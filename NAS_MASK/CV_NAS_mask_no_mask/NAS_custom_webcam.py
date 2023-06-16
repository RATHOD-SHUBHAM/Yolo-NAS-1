import cv2
import torch
from super_gradients.training import models
from super_gradients.common.object_names import Models

model = models.get('yolo_nas_l', num_classes= 3, checkpoint_path='weights/ckpt_best.pth')

model = model.to("cuda" if torch.cuda.is_available() else 'cpu')

# adding confidence score will show only those object whose which have confident score above that in %
# By default conf score is 0.25
model.predict_webcam()