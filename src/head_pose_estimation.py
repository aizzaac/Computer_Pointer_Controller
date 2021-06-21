import os
import logging
import cv2
from openvino.inference_engine import IECore

class HeadPoseEstimationClass():
    def __init__(self):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None



    def load_model(self, model, device = 'CPU', cpu_extensions = None):
        '''
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        model_xml = model #model's structure
        model_bin = os.path.splitext(model_xml)[0] + ".bin" #model's weights

        ### Load the model ###
        self.plugin = IECore() #Initialize the plugin (Inference Engine API)
        self.network = self.plugin.read_network(model = model_xml, weights = model_bin) #Read the IR as a IENetwork

        ### Add any necessary extensions ###
        if cpu_extensions is not None:
            self.plugin.add_extension(cpu_extensions, device)
     
        ### Check for supported layers ###
        supported_layers = self.plugin.query_network(network = self.network, device_name = device)
        
        unsupported_layers = [layer for layer in self.network.layers.keys() if layer not in supported_layers]
        if len(unsupported_layers) != 0:
            logging.error("Unsupported layers found: {}".format(unsupported_layers))
            logging.error("Check whether extensions are available to add to IECore.")
            exit(1)
         
        ### Load the network into the Inference Engine and return it as an Executable network ###
        self.exec_network = self.plugin.load_network(self.network, device_name = device, num_requests = 1)

        #Get the input of the network
        self.input_blob = next(iter(self.network.inputs))
        
        #Get the output of the network
        self.output_blob = next(iter(self.network.outputs))
        
        return
    


    def preprocess_input(self, image):
            '''
            Before feeding the data into the model for inference, it might need to be preprocessed, 
            according to what the network (model) accepts.
            Input (what the network accepts=what this function retuns)
            - shape:[1x3x384x672]
            - format:[BxCxHxW] = [Batch, Channels, Height, Width]
    
            *Expected color order: BGR
    
            -Image: H,W,C (default)
            -cv2.resize: (image, horizontal axis (width), vertical axis(height))
            '''
            net_input_shape = self.network.inputs[self.input_blob].shape #Gets the input shape of the network
    
            #Extract positions #3(width) and #2(height) and resize input image using those values.
            p_image = cv2.resize(image, (net_input_shape[3], net_input_shape[2]))#(image,(w,h))
            p_image = p_image.transpose((2, 0, 1))#change order of values: (h,w,3) -> (3,h,w)
            #Adding batch and collenting all the positional arguments in a tuple
            p_image = p_image.reshape(1, *p_image.shape)#(batch_size, channels, height, width) = (1,3,h,w)
            return p_image #returns preprocessed image [BxCxHxW]




    def predict(self, p_image):
        '''
        This method is meant for running predictions on the preprocessd image.
        Inputs
        - shape:[1x3x384x672]
        - format:[BxCxHxW] = [Batch, Channels, Height, Width]

        *Expected color order: BGR

        Outputs
        - shape:[1,1,N,7], N = #of detected bounding boxes
        - format:[image_id,label,conf,x_min,y_min,x_max,y_max]
          - image_id: ID of the image in the batch
          - label: predicted class ID (1-face)
          - conf: confidence for the predicted class
          - (x_min,y_min): coordinates of the top left bounding box corner
          - (x_max,y_max): coordinates of the bottom right bounding box corner

        - wait(-1) = time.sleep(1)

        '''
        self.exec_network.start_async(request_id=0, inputs={self.input_blob: p_image})#make async request

        if self.exec_network.requests[0].wait(-1) == 0:#wait for the async request to be completed
            outputs = self.exec_network.requests[0].outputs #extract the outputs
            return outputs#[1,1,N,7]



    def preprocess_output(self, outputs, image, args):
        '''
        Before feeding the output of this model to the next model, you might have to preprocess the output.
        - outputs: this is the array returned by the 'predict' function.
        - image: input image (without preprocessing).

        Outputs
        1. name: "angle_y_fc", shape: [1,1] - Estimated yaw (in degrees)
        2. name: "angle_p_fc", shape: [1,1] - Estimated pitch (in degrees)
        3. name: "angle_r_fc", shape: [1,1] - Estimated roll (in degrees)
        *Each output contains one float value that represents value in Tait-Bryan angles (yaw, pitch or roll)

        cv2.putText(image, text, coords, font, fontScale, color)
        '''
        yaw = outputs['angle_y_fc'][0][0] 
        pitch = outputs['angle_p_fc'][0][0]
        roll = outputs['angle_r_fc'][0][0]

        if args.view == "YES":#if user wants to preview the 'Text', etc
            cv2.putText(image, "YAW:{:.1f}".format(yaw), (20,30), 0, 1, (0, 0, 255)) #text in red
            cv2.putText(image, "PITCH:{:.1f}".format(pitch), (20,60), 0, 1, (0, 0, 255))
            cv2.putText(image, "ROLL:{:.1f}".format(roll), (20,90), 0, 1, (0, 0, 255))
        
        return image, [yaw, pitch, roll]
