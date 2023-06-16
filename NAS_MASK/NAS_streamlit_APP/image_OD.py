import cv2
import torch
from super_gradients.training import models
import numpy as np
import math

# get the model
device = 'cuda' if torch.cuda.is_available() else "cpu"
model = models.get('yolo_nas_l', num_classes= 3, checkpoint_path='weights/ckpt_best.pth')

# define class name
class_names = ['mask_worn_incorrect','with_mask', 'without_mask']

# Open CV to read image
image = cv2.imread('/Users/shubhamrathod/PycharmProjects/nas_streamlit/input/images/0_10725_jpg.rf.e5fdb8b47de7326a9e070beffc994f8a.jpg', cv2.IMREAD_UNCHANGED)


# Todo: Make Prediction
result = list(model.to(device).predict(image, conf = 0.65))[0]

# get  bounding box coordinates. confidence, labels

# bounding box coordinates
bbox_xyxy_cord = result.prediction.bboxes_xyxy.tolist()

# confidence score
confidence_score = result.prediction.confidence

# label
class_labels = result.prediction.labels.tolist()

for (bbox_xyxy, confidence, cls) in zip(bbox_xyxy_cord, confidence_score, class_labels):
    bbox = np.array(bbox_xyxy)
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

    # convert to integer
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # confidence
    conf = math.ceil((confidence * 100)) / 100

    # label
    class_integer = int(cls)
    # class_integer = 0 -> mask_worn_incorrect ,1 -> 'with_mask', 2 - > 'without_mask
    cls_name = class_names[class_integer]

    # combine class name and confidence score
    label = f'{cls_name}{","}{conf}'
    print(label)

    # text size of label and confidence
    # https://answers.opencv.org/question/7947/calculate-font-scale-to-fit-text-in-the-box/
    font = 0
    fontScale = 0.6
    thickness = 1
    # t_size, _ = cv2.getTextSize('Test', font, fontScale, thickness)
    t_size = cv2.getTextSize(label, font, fontScale, thickness)[0]
    weight_height = x1 + t_size[0] , y1 - t_size[1] - 3

    # display the label box and confidence
    # label and confidence score
    cv2.rectangle(image, (x1, y1), weight_height, [255, 0, 255], -1, cv2.LINE_AA) # filled
    cv2.putText(image, label, (x1, y1 - 2), 0, 0.6, [0, 0, 0], thickness = 1, lineType = cv2.LINE_AA)
    # bounding box
    bb_box_color = (255, 255, 255)
    bb_box_thickness = 1
    cv2.rectangle(image, (x1, y1), (x2, y2), bb_box_color, bb_box_thickness)

# resize
# resized_image = cv2.resize(image, (0,0), fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)

# save image
status = cv2.imwrite('/Users/shubhamrathod/PycharmProjects/nas_streamlit/output/image.png',image)

# Show Image
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()