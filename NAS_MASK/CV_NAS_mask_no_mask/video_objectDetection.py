import cv2
import torch
from super_gradients.training import models
import numpy as np
import math


device = 'cuda' if torch.cuda.is_available() else "cpu"
model = models.get('yolo_nas_l', num_classes= 3, checkpoint_path='weights/ckpt_best.pth')

class_names = ['mask_weared_incorrect','with_mask', 'without_mask']

# capture video
# cap = cv2.VideoCapture("/Users/shubhamrathod/PycharmProjects/CV_NAS_mask_no_mask/input/video/stock-footage-african-ethnic-woman-takes-off-her-protective-mask-smiles-breaths-in-deeply-rejoices-at-end-of.mp4")

# web cam
cap = cv2.VideoCapture(0)

# if video not available
if (cap.isOpened()==False):
    print("Unable to read the video")

# video writer
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('/Users/shubhamrathod/PycharmProjects/CV_NAS_mask_no_mask/output/Output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (frame_width, frame_height))

frame_count = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    frame_count += 1

    if ret:
        result = list(model.to(device).predict(frame, conf = 0.45))[0]

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
            # print("Frame Count: ", frame_count, "coordinates: " , x1, y1, x2, y2)

            # convert to integer
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print("Frame Count: ", frame_count, "coordinates: ", x1, y1, x2, y2)

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
            fontScale = 1.0
            thickness = 2
            # t_size, _ = cv2.getTextSize('Test', font, fontScale, thickness)
            t_size = cv2.getTextSize(label, font, fontScale, thickness)[0]
            weight_height = x1 + t_size[0] , y1 - t_size[1]

            # display the label box and confidence
            # label and confidence score
            cv2.rectangle(frame, (x1, y1), weight_height, [225, 0, 255], -1, cv2.LINE_AA) # filled
            cv2.putText(frame, label, (x1, y1), 0, 1, [225, 255, 255], thickness = 1, lineType = cv2.LINE_AA)
            # bounding box
            bb_box_color = (255, 255, 255)
            bb_box_thickness = 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), bb_box_color, bb_box_thickness)

            out.write(frame)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF==ord('1'):
            break
    else:
        break
#
#
# release the frame
out.release()
cap.release()
cv2.destroyAllWindows()
