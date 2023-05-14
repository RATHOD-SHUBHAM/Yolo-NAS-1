Here's a step-by-step breakdown of how I did it:

Data Collection: The first step was to collect a set of images relevant to the object detection task. I used RoboFlow's custom dataset for this task.

https://lnkd.in/g-NG597q
Data Augmentation or Data Preprocessing: After collecting the images, I cleaned and prepared the data for the training phase. This involved resizing, cropping, and labeling the images with bounding boxes to indicate the location of the objects of interest.

Model Training: With the preprocessed data ready, I used Yolo NAS to train the object detection model. I fine-tuned the pre-trained model and trained it on my custom dataset. I ran multiple iterations of the training process, adjusting the hyperparameters and monitoring the model's performance.

Model Evaluation: Once the model was trained, I evaluated its performance on a separate validation set to see how well it could detect objects in new, unseen images. The evaluation metrics I used included precision, recall, and F1 score.

Model Deployment: The final step will be to containerize and deploy it in an environment such as on-premises or the cloud.

The model output was an object detection system capable of identifying all 23 classes within images, which include classes like door, front bumper, headlamp, etc., with remarkable accuracy and speed. The system can detect multiple objects in a single image and provide a bounding box around each object.

This technology has tremendous potential for various applications, from autonomous driving to video surveillance and beyond.

<img width="1434" alt="Screenshot 2023-05-14 at 9 28 37 AM" src="https://github.com/RATHOD-SHUBHAM/OOD_YOLONAS_SAM/assets/58945964/962343e4-d7ec-4cd8-9c91-31994a98b2ec">
