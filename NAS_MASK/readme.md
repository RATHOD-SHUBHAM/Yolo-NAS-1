# YOLO_NAS - openCV - Streamlit

## End to End PipeLine

Here's a step-by-step breakdown of how I did it:

1.	Dataset Collection: The initial stage was to gather a collection of images related to the object detection task. For this challenge, I utilized RoboFlow's public dataset.\
[dataset](https://universe.roboflow.com/xml-to-yolo-sqqvs/face-masks-old-data)

2. Data Augmentation or Data Preprocessing: Following the collection of images, the data was cleaned and prepared for the training phase. This included scaling, cropping, and labeling the images with bounding boxes that indicated where the objects of interest were located.

3. Model Training: Using the Yolo NAS architecture, I trained the model on the custom dataset. Leveraging transfer learning, I fine-tuned the pre-trained model's weights to adapt it to my specific object detection task. I iteratively optimized the model's performance, adjusting hyperparameters and monitoring its progress.

4. Model Evaluation: Once training was complete, I evaluated the model's performance on a separate validation set to assess its accuracy and robustness. Evaluation metrics such as precision, recall, and F1 score were calculated to gauge its object detection capabilities.

5. Model Export and Conversion: After achieving satisfactory performance, I exported the trained model in a format.

6. Streamlit Web Application: To provide a user-friendly interface for interacting with the model, I leveraged Streamlit, a Python library for building web applications. I created a Streamlit app that allowed users to upload images and videos or perform real-time inference, and the model would perform object detection on those, highlighting and labeling the detected objects.

7. Deployment on Hugging Face: Once the model was complete, I used Hugging Face's Model Hub to upload and share it with the community. The application is available to users all over the world, allowing them to use the object detection capabilities effortlessly through a web browser.

I'm quite pleased with this work and the implications it has for businesses such as autonomous systems, security, and healthcare. 

If you're interested, you can check out the web application and witness the Yolo NAS concept in action.! \
[Mask_Det](https://huggingface.co/spaces/Rathsam/FaceMaskDetection_YOLONAS)

Code Available on GitHub: \
[Code](https://github.com/RATHOD-SHUBHAM/OOD_YOLONAS_SAM/tree/master/NAS_MASK/NAS_streamlit_APP)


![20dc6883cc5e6dac0dc599771f263c7e3e0697ef1b20cb97843a13a9](https://github.com/RATHOD-SHUBHAM/OOD_YOLONAS_SAM/assets/58945964/d3611399-bbd9-43c2-9a2a-767622576604)
