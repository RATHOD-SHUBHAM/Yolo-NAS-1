# YOLO_NAS + openCV + Streamlit

## End to End PipeLine

Here's a step-by-step breakdown of how I did it:

1.	Dataset Collection: 
  - The initial stage was to gather a collection of images related to the object detection task. For this challenge, I utilized RoboFlow's public dataset.\
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

Git Folder: [NAS_streamlit_APP](https://github.com/RATHOD-SHUBHAM/OOD_YOLONAS_SAM/tree/master/NAS_MASK/NAS_streamlit_APP)

![20dc6883cc5e6dac0dc599771f263c7e3e0697ef1b20cb97843a13a9](https://github.com/RATHOD-SHUBHAM/OOD_YOLONAS_SAM/assets/58945964/d3611399-bbd9-43c2-9a2a-767622576604)

---

## OpenCV + YOLO-NAS

To use OpenCV to get bounding box labels and confidence from a YOLO (You Only Look Once) model, you'll need to perform the following steps:

1. Install OpenCV and YOLO model: Ensure that you have OpenCV and the YOLO model installed on your system. You can install OpenCV using the following command:

   ```
   pip install opencv-python
   ```


2. Load the YOLO model in OpenCV
    ```
    model = models.get('yolo_nas_l', num_classes= 3, checkpoint_path='weights/ckpt_best.pth')
    ```

3. Read and preprocess the input image: 
  - Use OpenCV's `cv2.imread()` function to read the input image from the file system. 
  - Preprocess the image by resizing it to the required dimensions and normalizing the pixel values. 

4. Perform object detection: 
  - Pass the preprocessed image through the YOLO model to perform object detection. 
  ```
  model.to(device).predict(frame, conf = 0.45)
  ```

5. Extract bounding box labels and confidence: 
  - Iterate over the detections and extract the bounding box coordinates, class labels, and confidence scores. 
  - You can filter the detections based on a certain confidence threshold to obtain more accurate results.

6. Display or save the output:
    ```
    out.write(frame)
    cv2.imshow("Frame", frame)
    ```
 
Git Folder : [CV_NAS_mask_no_mask](https://github.com/RATHOD-SHUBHAM/OOD_YOLONAS_SAM/tree/master/NAS_MASK/CV_NAS_mask_no_mask)

---

## Working with this Repo

Here's a step-by-step guide to clone and run this repository:


1. Clone the repository: 
  - Open a terminal or command prompt and navigate to the directory where you want to clone the repository. 
  - Use the following command to clone the repository:
   ```
   git clone https://github.com/RATHOD-SHUBHAM/OOD_YOLONAS_SAM.git
   ```

2. Create a virtual environment: 
  - It's recommended to create a virtual environment to isolate the application's dependencies. 
  - Navigate to the cloned repository's directory and run the following commands:

   ```
   python -m venv myenv      # Create a virtual environment named "myenv"
   source myenv/bin/activate   # Activate the virtual environment (for Unix/Linux)
   myenv\Scripts\activate      # Activate the virtual environment (for Windows)
   ```

3. Install dependencies: 
  - Once inside the virtual environment, you need to install the required dependencies. 
  - Typically, the dependencies are listed in a `requirements.txt` file. 
  - Run the following command to install them:

   ```
   pip install -r requirements.txt
   ```

4. Run the Streamlit application: 
  - With the dependencies installed, you can now run the Streamlit application. 
  - Use the following command:

   ```
   streamlit run streamlit_app.py
   ```

   Replace `app.py` with the actual name of the main Python file that contains the Streamlit application code.

5. Access the application: 
  - After running the command, Streamlit will start a local server and provide a local URL (usually http://localhost:8501) where you can access the application. 
  - Open a web browser and visit the provided URL to see and interact with the application.


https://github.com/RATHOD-SHUBHAM/OOD_YOLONAS_SAM/assets/58945964/bdf7b0a7-2043-41ac-a1aa-742630fd76af

---

## Model Weight And Improving Accuracy

To improve performance, you could iterate through these steps:

1. Collect data: Increase the number of training examples.
2. Epoch: Train the model on a larger Epoch. (Here the model is trained on 10 Epoch).
3. Model parameter tuning: Consider tuning values for the training parameters used by  learning algorithm.

Model Weight: [Weight_File](https://drive.google.com/drive/u/5/folders/16qYHDYxRE1HQSLfydgtEXHbxrCOUhjoX)
---
