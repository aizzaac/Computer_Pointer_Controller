import os
import sys
import time
import cv2
import logging as log

from argparse import ArgumentParser

from input_feeder import InputFeeder

from face_detection import FaceDetectionClass
from facial_landmarks_regression import FacialLandmarksRegressionClass
from head_pose_estimation import HeadPoseEstimationClass
from gaze_estimation import GazeEstimationClass

from mouse_controller import MouseController



def build_argparser():
    '''
    Parse command line arguments.
    '''
    parser = ArgumentParser()

    parser.add_argument("-fd", "--fd_model_file", type=str, required=True,
                        help="Path of xml file of Face Detection model")

    parser.add_argument("-fl", "--flr_model_file", type=str, required=True,
                        help="Path of xml file of Facial Landmarks regression model")

    parser.add_argument("-hp", "--hpe_model_file", type=str, required=True,
                        help="Path of xml file of Head Pose Estimation model")

    parser.add_argument("-ge", "--ge_model_file", type=str, required=True,
                        help="Path of xml file of Gaze Estimation model")

    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to image or video file")

    parser.add_argument("-l", "--cpu_extension", required=False, type=str, default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")

    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable.")

    parser.add_argument("-pt", "--prob_threshold", required=False, type=float, default=0.6,
                        help="Probability threshold for detections filtering"
                        "(0.6 by default)")

    parser.add_argument("-v", "--view", type=str, required=False, default="NO",
                            help="To view models' outputs."
                            "(YES or NO is acceptable)")

    parser.add_argument("-o", "--output_path", default='result/', type=str,
                        help="Output video path")                   

    return parser
 
    

def build_logrecord():
    '''
    Display LogRecord as: 
    - Human-readable time when the log was created
    - Level for the message
    - Source line number where the logging was issued
    - The logged message 

    *root logger level: INFO
    *Handlers: send the LogRecord to the console and to a file
    '''
    log.basicConfig(level = log.INFO,
                    format = "%(asctime)s [%(levelname)s] %(lineno)d %(message)s", datefmt='%d-%b-%y %H:%M:%S',
                    handlers = [log.FileHandler("Computer_Pointer_Controller.log"), log.StreamHandler()])



def infer_on_stream(args):
   
    output_path = args.output_path#path to store the results
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    #MouseController: precision="low", speed="fast"
    mouse_control = MouseController("low", "fast")
    
    
    #Media type: video, image or cam
    i_feeder = InputFeeder('video', args.input)
    i_feeder.load_data()
    
    #1920x1080
    i_width = int(i_feeder.cap.get(cv2.CAP_PROP_FRAME_WIDTH))#cap.get(3)
    i_height = int(i_feeder.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))#cap.get(4)
    
    FPS = int(i_feeder.cap.get(cv2.CAP_PROP_FPS)/10)

    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), FPS, (i_width, i_height), True)



    try:
        log.info("############# Looad Time #############")
        start_total_load_time = time.monotonic()

        #Face Detection model load time in ms
        start_time = time.monotonic()
        plugin_fd = FaceDetectionClass()#Initialize FaceDetection class
        plugin_fd.load_model(args.fd_model_file, args.device, args.cpu_extension)
        log.info("Face Detection model (ms): {:.1f}".format(1000 * (time.monotonic() - start_time)))


        #Head Pose Estimation model load time in ms
        start_time = time.monotonic()
        plugin_hpe = HeadPoseEstimationClass()
        plugin_hpe.load_model(args.hpe_model_file, args.device, args.cpu_extension)
        log.info("Head Pose Estimation model (ms): {:.1f}".format(1000 * (time.monotonic() - start_time)))


        #Facial Landmarks Regression model load time in ms
        start_time = time.monotonic()
        plugin_flr = FacialLandmarksRegressionClass()
        plugin_flr.load_model(args.flr_model_file, args.device, args.cpu_extension)
        log.info("Facial Landmarks Regression model (ms): {:.1f}".format(1000 * (time.monotonic() - start_time)))


        #Gaze Estimation model load time in ms
        start_time = time.monotonic()
        plugin_ge = GazeEstimationClass()
        plugin_ge.load_model(args.ge_model_file, args.device, args.cpu_extension)
        log.info("Gaze Estimation model(ms): {:.1f}".format(1000 * (time.monotonic() - start_time)))
        


        total_model_load_time = time.monotonic() - start_total_load_time
        log.info("##########################################")
                 
    except Exception as e:
        log.error("ERROR in loading models: " + str(e))
        sys.exit(1)



    #Timers and counter
    start_total_inference_time = time.monotonic()#all models

    fd_inference_time = 0 #face_detection
    hpe_inference_time = 0#head_position_estimation
    flr_inference_time = 0#facial_landmarks_regression
    ge_inference_time = 0#gaze_estimation
    
    frame_counter = 0#frame counter



    while True:
        try:
            frame = next(i_feeder.next_batch())
        except StopIteration:
            break

        key_pressed = cv2.waitKey(60)
        
        
        frame_counter += 1 #increase by 1

        ## Face Detecton Model
        p_image = plugin_fd.preprocess_input(frame)

        start_time = time.monotonic()
        outputs = plugin_fd.predict(p_image)
        fd_inference_time += (time.monotonic() - start_time)
        
        out_frame, faces = plugin_fd.preprocess_output(outputs, frame, args)
        

        for face in faces:
            crop_face = frame[face[1]:face[3], face[0]:face[2]]


            ## Head Pose Estimation Model
            p_crop_face = plugin_hpe.preprocess_input(crop_face)

            start_time = time.monotonic()
            outputs = plugin_hpe.predict(p_crop_face)
            hpe_inference_time += (time.monotonic() - start_time)
            out_frame, out_angles = plugin_hpe.preprocess_output(outputs, out_frame, args)


            ## Facial Landmarks Detecton Model
            p_crop_face = plugin_flr.preprocess_input(crop_face)

            start_time = time.monotonic()
            outputs = plugin_flr.predict(p_crop_face)
            flr_inference_time += (time.monotonic() - start_time)
            crop_left_eye, crop_right_eye, left_eye_point, right_eye_point = plugin_flr.preprocess_output(outputs, crop_face, args)##??


            ## Gaze Estimation Model
            out_frame, left_eye, right_eye  = plugin_ge.preprocess_input(out_frame, crop_left_eye, crop_right_eye)#??

            start_time = time.monotonic()
            outputs = plugin_ge.predict(left_eye, right_eye, out_angles)
            ge_inference_time += (time.monotonic() - start_time)
            out_frame, gazevector = plugin_ge.preprocess_output(outputs, out_frame, crop_face, left_eye_point, right_eye_point, args)

            cv2.imshow("Computer Pointer Control", out_frame)
            out_video.write(out_frame)
            mouse_control.move(gazevector[0], gazevector[1])


        if key_pressed == 27:
            break


    if frame_counter > 0:
        log.info("############# Models Inference Time #############")
        log.info("Face Detection Model (ms): {:.1f}".format(1000 * fd_inference_time / frame_counter))
        log.info("Facial Landmarks Detection Model (ms): {:.1f}".format(1000 * flr_inference_time / frame_counter))
        log.info("Head Pose Detection Model (ms): {:.1f}".format(1000 * hpe_inference_time / frame_counter))
        log.info("Gaze Detection Model (ms): {:.1f}".format(1000 * ge_inference_time / frame_counter))
        log.info("##########################################")


    #total_infer_time = time.monotonic() - start_total_inference_time
    total_inference_time = round((time.monotonic() - start_total_inference_time),1)
    fps = frame_counter / total_inference_time
    
    
    #Save statistics 
    with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
        f.write(str(total_model_load_time)+'\n')
        f.write(str(total_inference_time)+'\n')
        f.write(str(fps)+'\n')
        


    log.info("############# Stats summary #############")
    log.info(f"Total Model Load Time: {total_model_load_time}")
    log.info(f"Total Inference Time: {total_inference_time}")
    log.info(f"FPS: {fps}")
    log.info("#################################################")



    #close the video capture
    i_feeder.close()
    cv2.destroyAllWindows()



def main():
    args = build_argparser().parse_args()
    build_logrecord()
    infer_on_stream(args)



if __name__ == '__main__':
    main()