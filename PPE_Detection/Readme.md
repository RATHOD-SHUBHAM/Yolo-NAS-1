# PPE detection using YOLO-NAS

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


https://github.com/RATHOD-SHUBHAM/OOD_YOLONAS_SAM/assets/58945964/de54a7e2-cab8-4c78-a4c9-f23deb8e27b4

