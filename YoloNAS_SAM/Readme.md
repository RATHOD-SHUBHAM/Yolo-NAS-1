# Yolo NAS + SAM

*Yolo NAS model is explored previously.*

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

Kaggle Notebook: https://www.kaggle.com/code/gibborathod/segmentanything?scriptVersionId=130225073
