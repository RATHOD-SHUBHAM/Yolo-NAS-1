# OOD_YOLONAS_SAM

## 1] YOLO_NAS on Custom DataSet:

Here's a step-by-step breakdown of how I did it:

* Data Collection: The first step was to collect a set of images relevant to the object detection task. I used RoboFlow's custom dataset for this task.
    ##### https://lnkd.in/g-NG597q

* Data Augmentation or Data Preprocessing: After collecting the images, I cleaned and prepared the data for the training phase. This involved resizing, cropping, and labeling the images with bounding boxes to indicate the location of the objects of interest.

* Model Training: With the preprocessed data ready, I used Yolo NAS to train the object detection model. I fine-tuned the pre-trained model and trained it on my custom dataset. I ran multiple iterations of the training process, adjusting the hyperparameters and monitoring the model's performance.

* Model Evaluation: Once the model was trained, I evaluated its performance on a separate validation set to see how well it could detect objects in new, unseen images. The evaluation metrics I used included precision, recall, and F1 score.

* Model Deployment: The final step will be to containerize and deploy it in an environment such as on-premises or the cloud.

The model output was an object detection system capable of identifying all 23 classes within images, which include classes like door, front bumper, headlamp, etc., with remarkable accuracy and speed. The system can detect multiple objects in a single image and provide a bounding box around each object.

This technology has tremendous potential for various applications, from autonomous driving to video surveillance and beyond.


<img width="1434" alt="Screenshot 2023-05-14 at 9 28 37 AM" src="https://github.com/RATHOD-SHUBHAM/OOD_YOLONAS_SAM/assets/58945964/962343e4-d7ec-4cd8-9c91-31994a98b2ec">

---

## 2. Yolo NAS + SAM

*Yolo NAS model is explored above.*

Steps:

1. Install the necessary libraries and frameworks: You will need to install libraries and frameworks like OpenCV, Supergradient, SAM which are required for object detection.

2. Download the YOLO NAS model: You can download the YOLO NAS model from the official website or from GitHub. This model is trained on the COCO dataset, which includes a large number of object classes.

3. Download the Segment Anything model: You can download the Segment Anything model from GitHub. This model is trained to segment objects from an image.

4. Load the YOLO NAS model: Use Keras to load the YOLO NAS model into your project.

5. Load the Segment Anything model: Use TensorFlow to load the Segment Anything model into your project.

6. Load the input image: Load the input image that you want to perform object detection on.

7. Perform object detection: Use the YOLO NAS model to detect objects in the input image. This will give you a list of bounding boxes and confidence scores for each object detected.

8. Segment the objects: Use the Segment Anything model to segment the objects detected in the input image.

9. Provide Bounding Box Coordinates obtained from YOLO NAS to SAM.

10. Visualize the results: Visualize the results of object detection and segmentation by drawing bounding boxes around the objects detected and coloring the segmented objects.

<img width="1733" alt="Screenshot 2023-05-14 at 2 38 54 PM" src="https://github.com/RATHOD-SHUBHAM/OOD_YOLONAS_SAM/assets/58945964/1b031f8a-10f7-4392-ad59-7304bbe3df6c">
    
---
## 3] PPE detection using YOLO-NAS

To perform PPE (Personal Protective Equipment) detection using YOLO-NAS (You Only Look Once Neural Architecture Search) and Roboflow data, you can follow these steps:

1. Collect and Prepare the Dataset:
   - Obtain a dataset that contains images labeled with bounding boxes for different PPE items (e.g., helmets, vests, goggles, masks).
   - Organize the dataset in a format compatible with YOLO-NAS.
   - I obtained that dataset from roboflow: https://universe.roboflow.com/objet-detect-yolov5/eep_detection-u9bbd

2. Set up the Environment:
   - Install the necessary dependencies, including Python, TensorFlow, and other required libraries.
   - Download and set up YOLO-NAS from the official repository or any other reliable source.
   - Make sure you have the Roboflow account and the necessary API key for data augmentation and preprocessing.

3. Data Preprocessing:
   - Use Roboflow's to preprocess and augment your dataset. 
   - Download the data or use Roboflow's API to download datset in yolov5 format. 

4. Model Training:
   - Use the training set to train the YOLO-NAS model. 
   - Fine-tune the model using transfer learning with a pre-trained model available. This step can improve the model's accuracy and speed up the training process.
   - Monitor the training process and evaluate the model's performance on the validation set. Adjust hyperparameters as needed.

5. Model Evaluation and Testing:
   - Once the model is trained, evaluate its performance using appropriate evaluation metrics like mean Average Precision (mAP).
   - Test the model on unseen images or a separate test set to assess its generalization capability and detect PPE items in real-world scenarios.

6. Model Deployment:
   - Export the trained YOLO-NAS model for inference.


<img width="464" alt="Screenshot 2023-05-30 at 5 59 41 PM" src="https://github.com/RATHOD-SHUBHAM/OOD_YOLONAS_SAM/assets/58945964/ea65431a-b7e8-483a-9313-10c98aa557d1">

---

## 4] YOLO-NAS + OpenCV + Streamlit

1.	Dataset Collection: 
    - The initial stage was to gather a collection of images related to the object detection task. For this challenge, I utilized RoboFlow's public dataset.
    - [dataset](https://universe.roboflow.com/xml-to-yolo-sqqvs/face-masks-old-data)

2. Data Augmentation or Data Preprocessing: 
    - Following the collection of images, the data was cleaned and prepared for the training phase. This included scaling, cropping, and labeling the images with bounding boxes that indicated where the objects of interest were located.

3. Model Training: 
    - Using the Yolo NAS architecture, I trained the model on the custom dataset. Leveraging transfer learning, I fine-tuned the pre-trained model's weights to adapt it to my specific object detection task. I iteratively optimized the model's performance, adjusting hyperparameters and monitoring its progress.

4. Model Evaluation: 
    - Once training was complete, I evaluated the model's performance on a separate validation set to assess its accuracy and robustness. Evaluation metrics such as precision, recall, and F1 score were calculated to gauge its object detection capabilities.

5. Model Export and Conversion: 
    - After achieving satisfactory performance, I exported the trained model in a format.

6. Streamlit Web Application: 
    - To provide a user-friendly interface for interacting with the model, I leveraged Streamlit, a Python library for building web applications. I created a Streamlit app that allowed users to upload images and videos or perform real-time inference, and the model would perform object detection on those, highlighting and labeling the detected objects.

7. Deployment on Hugging Face: 
    - Once the model was complete, I used Hugging Face's Model Hub to upload and share it with the community. The application is available to users all over the world, allowing them to use the object detection capabilities effortlessly through a web browser.

Live Demo: [Mask_Det](https://huggingface.co/spaces/Rathsam/FaceMaskDetection_YOLONAS)

<img width="732" alt="Screenshot 2023-06-15 at 9 59 03 PM" src="https://github.com/RATHOD-SHUBHAM/OOD_YOLONAS_SAM/assets/58945964/9d332b0c-2c4e-4efe-b710-bafdaf9da894">

---


