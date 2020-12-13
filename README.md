# flask_server_ml_display
This uses FRCNN model as object detection and a flask server to display the images of the prediction. The weigh of the FRCNN is read with OpenCV. This is an incremental work and there is a lot to be done with the database aspect of the project and other features. You are free to download it and use the project as you wish. <br/>

# Requirements. 
(1) You have to wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz  Extract it and add the frozen_inference_graph.pb in the FRCNN folder. for the project to work. <br/>
(2) Install flask. <br/> 
(3) Install OpenCV. <br/> 
(4) Matplotlib. <br/> 
(5) Numpy. <br/> 

# Usage
(1) Open a terminal and enter python main.py to start up the flask server. <br/>
(2) Open a browser and type http://127.0.0.1:9100 to display the page. <br/>
(3) Click the browse button to select your desired image to be predicted. <br/>
(4) Click upload button to upload the image and render the annotated image with appropriate bounding box classes. <br/>


![alt text](https://github.com/adezoguns/flask_server_ml_display/blob/main/screenshot.png)

