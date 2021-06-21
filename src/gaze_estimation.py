import os
import logging
import cv2
from openvino.inference_engine import IECore


class GazeEstimationClass():   
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
    
    
    

    def preprocess_input(self, frame, crop_left_eye, crop_right_eye):
        '''
        Before feeding the data into the model for inference, it might need to be preprocessed, 
        according to what the network (model) accepts.

        Input (what the network accepts=what this function retuns)
        - square crop of left eye image
          * shape:[1x3x60x60]
          * format:[BxCxHxW] = [Batch, Channels, Height, Width]
          * name:left_eye_image
        - square crop of right eye image
          * shape:[1x3x60x60]
          * format:[BxCxHxW] = [Batch, Channels, Height, Width]
          * name:right_eye_image
        - 3 head pose angles (yaw, pitch and roll)
          * shape:[1x3]
          * format:[BxC]
          * name:head_pose_angles

        *Expected color order: BGR

        -Image: H,W,C (default)
        -cv2.resize: (image, horizontal axis(width), vertical axis(height))
        '''
        left_net_input_shape = self.network.inputs['left_eye_image'].shape#Gets the input shape of the network (left eye)

        #Extract positions #3(width) and #2(height) and resize input image using those values.
        p_left_eye = cv2.resize(crop_left_eye, (left_net_input_shape[3], left_net_input_shape[2]))#(image,(w,h))
        p_left_eye = p_left_eye.transpose((2, 0, 1))#change order of values: (h,w,3) -> (3,h,w)
        #Adding batch and collenting all the positional arguments in a tuple
        p_left_eye = p_left_eye.reshape(1, *p_left_eye.shape)#(batch_size, channels, height, width) = (1,3,h,w)
        
        
        
        right_net_input_shape = self.network.inputs['right_eye_image'].shape#Gets the input shape of the network (right eye)

        #Extract positions #3(width) and #2(height) and resize input image using those values.
        p_right_eye = cv2.resize(crop_right_eye, (right_net_input_shape[3], right_net_input_shape[2]))#(image,(w,h))
        p_right_eye = p_right_eye.transpose((2, 0, 1))#change order of values: (h,w,3) -> (3,h,w)
        #Adding batch and collenting all the positional arguments in a tuple
        p_right_eye = p_right_eye.reshape(1, *p_right_eye.shape)#(batch_size, channels, height, width) = (1,3,h,w)
        

        return frame, p_left_eye, p_right_eye



    

    def predict(self, left_eye_image, right_eye_image, headpose_angles):
        '''
        This method is meant for running predictions on the preprocessed image.

        Input (what the network accepts=what this function retuns)
        - square crop of left eye image
          * shape:[1x3x60x60]
          * format:[BxCxHxW] = [Batch, Channels, Height, Width]
          * name:left_eye_image
        - square crop of right eye image
          * shape:[1x3x60x60]
          * format:[BxCxHxW] = [Batch, Channels, Height, Width]
          * name:right_eye_image
        - 3 head pose angles (yaw, pitch and roll)
          * shape:[1x3]
          * format:[BxC]
          * name:head_pose_angles

        Output
        - Cartesian coordinates of gaze direction vector (x,y,z)
          * shape:[1,3]
          * name:gaze_vector
         Output vector is not normalized and has non-unit length

        wait(-1) = time.sleep(1)
        '''
        self.exec_network.start_async(request_id=0, inputs = {'left_eye_image': left_eye_image, 
                                                              'right_eye_image': right_eye_image,
                                                              'head_pose_angles': headpose_angles})
        if self.exec_network.requests[0].wait(-1) == 0:#wait for the async request to be completed
            outputs = self.exec_network.requests[0].outputs[self.output_blob]#extract the outputs
            return outputs#[1,3]






    def preprocess_output(self, outputs, image, crop_face, left_eye_point, right_eye_point, args):
        '''
        Before feeding the output of this model to the next model, you might have to preprocess the output.
        Output
        - Cartesin coordinates of gaze direction vector (x,y,z)
          * shape:[1,3]
          * name:gaze_vector
         Output vector is not normalized and has non-unit length

         cv2.arrowedLine(image, start_point, end_point, color)
         cv2.putText(image, text, coords, font, fontScale, color)
        '''
        x = outputs[0][0]
        y = outputs[0][1]
        z = outputs[0][2]
        
        
        #Endpoint coords for arrow
        xlmax = int(left_eye_point[0]+x*200)
        ylmin = int(left_eye_point[1]-y*200)
        xrmax = int(right_eye_point[0]+x*200)
        yrmin = int(right_eye_point[1]-y*200)


        if args.view == "YES":#if user wants to preview the 'bounding boxes', etc
            cv2.putText(image, "X:{:.1f}".format(x * 100), (20, 150), 0, 1, (0, 0, 0))#black
            cv2.putText(image, "Y:{:.1f}".format(y * 100), (20, 190), 0, 1, (0, 0, 0))
            cv2.putText(image, "Z:{:.1f}".format(z), (20, 230), 0, 1, (0, 0, 0))


            cv2.arrowedLine(crop_face, left_eye_point, (xlmax, ylmin), (0,0,0), 2)#black arrow
            cv2.arrowedLine(crop_face, right_eye_point, (xrmax, yrmin), (0,0,0), 2)#black arrow

        return image, [x, y, z]#only "x" and "y" because computer screen is 2D.