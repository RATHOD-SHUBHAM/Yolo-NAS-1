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

[https://github.com/RATHOD-SHUBHAM/OOD_YOLONAS_SAM/assets/58945964/de54a7e2-cab8-4c78-a4c9-f23deb8e27b4](https://github.com/RATHOD-SHUBHAM/OOD_YOLONAS_SAM/assets/58945964/2e26baa6-143d-4d13-9564-c2752ffd3c65)


