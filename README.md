# Computer Pointer Controller

This is a computer vision app that moves the mouse pointer using the user's gaze. Four Intel pre-trained models have been used:
* face-detection-adas-0001
* head-pose-estimation-adas-0001
* landmarks-regression-retail-0009
* gaze-estimation-adas-0002

In this project, the models execute in sequence: the output of a model is used as the input of other model. 


## Project Set Up and Installation
1. Download and install OpenVINO
   * [OpenVINO 2020.2](https://docs.openvinotoolkit.org/2020.2/_docs_install_guides_installing_openvino_linux.html#install-openvino)


2. Download the required models
   * [face-detection](https://docs.openvinotoolkit.org/latest/omz_models_model_face_detection_adas_0001.html)

   * [head-pose](https://docs.openvinotoolkit.org/latest/omz_models_model_head_pose_estimation_adas_0001.html)

   * [facial-landmarks](https://docs.openvinotoolkit.org/latest/omz_models_model_landmarks_regression_retail_0009.html)

   * [gaze-estimation](https://docs.openvinotoolkit.org/latest/omz_models_model_gaze_estimation_adas_0002.html)


3. Create a virtual environment
   ```python3 -m venv myenv```
   ```source chico/bin/activate```


4. Set OpenVINO environment variables
   ```source /opt/intel/openvino_2020.2/bin/setupvars.sh```


5. Install dependecies
   ```pip3 install -r requirements.txt```


6. Execute main.py script
   The precisions can be changed: FP32, FP16, FP16-INT8

   ```python3 src/main.py -fd models/face-detection-adas-0001/FP32/face-detection-adas-0001.xml -hp models/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml -fl models/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml -ge models/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml -i bin/demo.mp4 -d "CPU" -o "result/FP32" -v "YES"```


## Documentation
1. Tree structure

```
├── images
│   ├── models.png
│   └── results_&_scripts.png
│   ├── FPS.png
│   └── Total_inference_time.png
│   │── Total_load_time.png
│   
|
├── models
│   ├── face-detection-adas-0001
│   │   ├── FP16
│   │   │   ├── face-detection-adas-0001.bin
│   │   │   └── face-detection-adas-0001.xml
│   │   ├── FP16-INT8
│   │   │   ├── face-detection-adas-0001.bin
│   │   │   └── face-detection-adas-0001.xml
│   │   └── FP32
│   │       ├── face-detection-adas-0001.bin
│   │       └── face-detection-adas-0001.xml
│   ├── gaze-estimation-adas-0002
│   │   ├── FP16
│   │   │   ├── gaze-estimation-adas-0002.bin
│   │   │   └── gaze-estimation-adas-0002.xml
│   │   ├── FP16-INT8
│   │   │   ├── gaze-estimation-adas-0002.bin
│   │   │   └── gaze-estimation-adas-0002.xml
│   │   └── FP32
│   │       ├── gaze-estimation-adas-0002.bin
│   │       └── gaze-estimation-adas-0002.xml
│   ├── head-pose-estimation-adas-0001
│   │   ├── FP16
│   │   │   ├── head-pose-estimation-adas-0001.bin
│   │   │   └── head-pose-estimation-adas-0001.xml
│   │   ├── FP16-INT8
│   │   │   ├── head-pose-estimation-adas-0001.bin
│   │   │   └── head-pose-estimation-adas-0001.xml
│   │   └── FP32
│   │       ├── head-pose-estimation-adas-0001.bin
│   │       └── head-pose-estimation-adas-0001.xml
│   └── landmarks-regression-retail-0009
│       ├── FP16
│       │   ├── landmarks-regression-retail-0009.bin
│       │   └── landmarks-regression-retail-0009.xml
│       ├── FP16-INT8
│       │   ├── landmarks-regression-retail-0009.bin
│       │   └── landmarks-regression-retail-0009.xml
│       └── FP32
│           ├── landmarks-regression-retail-0009.bin
│           └── landmarks-regression-retail-0009.xml
├── requirements.txt
├── result
│   ├── FP16
│   │   └── stats.txt
│   ├── FP16-INT8
│   │   └── stats.txt
│   └── FP32
│       └── stats.txt
└── src
│   ├── face_detection.py
│   ├── facial_landmarks_regression.py
│   ├── gaze_estimation.py
│   ├── head_pose_estimation.py
│   ├── input_feeder.py
│   ├── main.py
│   └── mouse_controller.py
│ 
│ 
│── Computer_Pointer_Controller_FP32.log 
|── Computer_Pointer_Controller_FP16.log  
│── Computer_Pointer_Controller_FP16-INT8.log
│── README.md  

```


2. Parsing arguments

```
-fd, --fd_model_file  --> Path of xml file of Face Detection model

-fl, --flr_model_file --> Path of xml file of Facial Landmarks regression model

-hp, --hpe_model_file --> Path of xml file of Head Pose Estimation model

-ge, --ge_model_file  --> Path of xml file of Gaze Estimation model

-i,  --input          --> Path to image or video file

-d,  --device         --> Specify the target device to infer on

-pt  --prob_threshold --> Probability threshold for detections filtering (default=0.6)

-v,  --view           --> To view models' outputs. YES or NO is acceptable

-o   --output_path    --> Output video path
```



## Benchmarks
For benchmarking, the following metrics were used:

* Total models loading time (sec)
* Total inference time (sec)
* Frames per second (FPS)

Some plots can be seen in Benchmark_CPC.ipynb or Benchmark_CPC.pdf.


| Precision  |  Model Load Time  | Inference Time |        FPS        |
|:-----------|:-----------------:|:--------------:|------------------:|
|   FP32     | 0.444525421000435 |       94.1     | 6.323060573857599 |
|   FP16     | 0.4293057909962954|      103.3     | 5.759922555663118 |
| FP16-INT8  | 0.8952202030050103|      111.0     | 5.36036036036036  |


The face_detection model is the one which uses more time for loading and for making inference, in the 3 precisions (FP32, FP16, FP16-INT8). It is the bottleneck. 

Values are similar when comparing the different precisions. The strongest difference is observed in the "Model Load Time" with FP16-INT8. Here, all the models use compressed models but still consume the double of time when compared to the other precisions. A possible explanation is that the code (software) is not fully optimized.

 


### Async Inference
With async inference request, the HW (CPU, FPGA, etc) is capable of doing more inferences at the same time because it is not required to wait for previous inferences to finish. This is very useful when many cameras will be working in simultaneous.
So, it can be said that this type of inference is good to  fully use the HW. 

Sync inferences is better suited for applications that require low latency (deterministic networks).



### Edge Cases
Most of time was used in installing openVINO. I had problems with the latest version (2021.3) and also when trying to do this project in the workspaces. That is why I decided to install an old version. But now that the application is working, I can explore newest versions of OpenVINO. Maybe my script will be faster.

Some edge cases that come to my mind are:
* when the camera detects many persons. which one will be able to control the mouse pointer?
* when there is not enough light and the camera believes that a person is present.
