import cv2
import torch
from super_gradients.training import models


device = 'cuda' if torch.cuda.is_available() else "cpu"
model = models.get('yolo_nas_l', num_classes= 3, checkpoint_path='weights/ckpt_best.pth')

# Open CV to read image
image = cv2.imread('/Users/shubhamrathod/PycharmProjects/CV_NAS_mask_no_mask/input/images/0_10725_jpg.rf.e5fdb8b47de7326a9e070beffc994f8a.jpg', cv2.IMREAD_UNCHANGED)



'''
# Resize the image if needed

scale_percent = 60  # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
'''

# adding confidence score will show only those object whose which have confident score above that in %
# By default conf score is 0.25
out = model.to(device).predict(image)

out.show()


