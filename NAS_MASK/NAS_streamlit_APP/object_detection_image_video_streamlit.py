import cv2
import torch
from super_gradients.training import models
import numpy as np
import math
import streamlit as st
import time



def load_yolonas_process_image(image,conf):
    # get the model
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    model = models.get('yolo_nas_l', num_classes= 3, checkpoint_path='weights/ckpt_best.pth')

    # define class name
    class_names = ['mask_worn_incorrect','with_mask', 'without_mask']

    # Open CV to read image
    # image = cv2.imread('/Users/shubhamrathod/PycharmProjects/nas_streamlit/input/images/0_10725_jpg.rf.e5fdb8b47de7326a9e070beffc994f8a.jpg', cv2.IMREAD_UNCHANGED)
    # image read from user
    image_col1, image_col2 = st.columns(2)
    im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with image_col1:
        st.subheader("Raw Image")
        st.image(im_rgb)


    # Todo: Make Prediction
    result = list(model.to(device).predict(image, conf = conf))[0]

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
        # print(label)

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
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    with image_col2:
        st.subheader('Inferred Image')
        st.image(image, channels= 'BGR', use_column_width=True)

# --------------------------------------------------------------------------------------------------------------------------------------------------------

def load_yolonas_process_video(video_name, v_conf, v_frame_rate, v_height, v_width, stContainer):
    # get the model
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    model = models.get('yolo_nas_l', num_classes=3, checkpoint_path='weights/ckpt_best.pth')

    # define class name
    class_names = ['mask_weared_incorrect', 'with_mask', 'without_mask']

    # capture video
    cap = cv2.VideoCapture(video_name)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # web cam
    # cap = cv2.VideoCapture(0)

    # Video writer
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('/Users/shubhamrathod/PycharmProjects/nas_streamlit/output/Output.avi',
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (frame_width, frame_height))

    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # out = cv2.VideoWriter('/Users/shubhamrathod/PycharmProjects/nas_streamlit/output/Output.mp4',fourcc, 200, (frame_width,frame_height))

    start_time = 0
    frame_count = 0

    while (cap.isOpened()):
        ret, frame = cap.read()
        frame_count += 1

        if ret:
            result = list(model.to(device).predict(frame, conf=v_conf))[0]

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

                # confidence
                conf = math.ceil((confidence * 100)) / 100

                # label
                class_integer = int(cls)
                # class_integer = 0 -> mask_worn_incorrect ,1 -> 'with_mask', 2 - > 'without_mask
                cls_name = class_names[class_integer]

                # combine class name and confidence score
                label = f'{cls_name}{","}{conf}'

                # text size of label and confidence
                # https://answers.opencv.org/question/7947/calculate-font-scale-to-fit-text-in-the-box/
                font = 0
                fontScale = 0.6
                thickness = 1
                # t_size, _ = cv2.getTextSize('Test', font, fontScale, thickness)
                t_size = cv2.getTextSize(label, font, fontScale, thickness)[0]
                weight_height = x1 + t_size[0], y1 - t_size[1] - 3

                # display the label box and confidence
                # label and confidence score
                cv2.rectangle(frame, (x1, y1), weight_height, [255, 0, 255], -1, cv2.LINE_AA)  # filled
                cv2.putText(frame, label, (x1, y1 - 2), 0, 0.6, [0, 0, 0], thickness=1, lineType=cv2.LINE_AA)
                # bounding box
                bb_box_color = (255, 255, 255)
                bb_box_thickness = 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), bb_box_color, bb_box_thickness)



            current_time = time.time()
            fps = 1 / (current_time - start_time)
            start_time =current_time

            v_frame_rate.write(f"<h1 style='text-align:center; color:red;'>{'{:.1f}'.format(fps)}</h1>",
                             unsafe_allow_html = True)
            v_height.write(f"<h1 style='text-align:center; color:red;'>{'{:.1f}'.format(video_width)}</h1>",
                             unsafe_allow_html=True)
            v_width.write(f"<h1 style='text-align:center; color:red;'>{'{:.1f}'.format(video_height)}</h1>",
                             unsafe_allow_html=True)

            stContainer.image(frame, channels='BGR', use_column_width=True)



            out.write(frame)



        else:
            break

    # release the frame
    st.warning("Infered video saved to : /Users/shubhamrathod/PycharmProjects/nas_streamlit/output")
    out.release()
    cap.release()
    cv2.destroyAllWindows()


# --------------------------------------------------------------------------------------------------------------------------------------------------------

def load_yolonas_process_RTI_video(rti_conf, rti_frame_rate, rti_height, rti_width, rti_stContainer):
    # get the model
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    model = models.get('yolo_nas_l', num_classes=3, checkpoint_path='weights/ckpt_best.pth')

    # define class name
    class_names = ['mask_weared_incorrect', 'with_mask', 'without_mask']

    # capture video
    # cap = cv2.VideoCapture(video_name)
    # video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # web cam
    cap = cv2.VideoCapture(0)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('/Users/shubhamrathod/PycharmProjects/nas_streamlit/output/Output.avi',
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (frame_width, frame_height))

    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # out = cv2.VideoWriter('/Users/shubhamrathod/PycharmProjects/nas_streamlit/output/Output.mp4',fourcc, 200, (frame_width,frame_height))

    start_time = 0
    frame_count = 0

    while (cap.isOpened()):
        ret, frame = cap.read()
        frame_count += 1

        if ret:
            result = list(model.to(device).predict(frame, conf=rti_conf))[0]

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

                # confidence
                conf = math.ceil((confidence * 100)) / 100

                # label
                class_integer = int(cls)
                # class_integer = 0 -> mask_worn_incorrect ,1 -> 'with_mask', 2 - > 'without_mask
                cls_name = class_names[class_integer]

                # combine class name and confidence score
                label = f'{cls_name}{","}{conf}'

                # text size of label and confidence
                # https://answers.opencv.org/question/7947/calculate-font-scale-to-fit-text-in-the-box/
                font = 0
                fontScale = 0.6
                thickness = 1
                # t_size, _ = cv2.getTextSize('Test', font, fontScale, thickness)
                t_size = cv2.getTextSize(label, font, fontScale, thickness)[0]
                weight_height = x1 + t_size[0], y1 - t_size[1] - 3

                # display the label box and confidence
                # label and confidence score
                cv2.rectangle(frame, (x1, y1), weight_height, [255, 0, 255], -1, cv2.LINE_AA)  # filled
                cv2.putText(frame, label, (x1, y1 - 2), 0, 0.6, [0, 0, 0], thickness=1, lineType=cv2.LINE_AA)
                # bounding box
                bb_box_color = (255, 255, 255)
                bb_box_thickness = 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), bb_box_color, bb_box_thickness)



            current_time = time.time()
            fps = 1 / (current_time - start_time)
            start_time =current_time

            rti_frame_rate.write(f"<h1 style='text-align:center; color:red;'>{'{:.1f}'.format(fps)}</h1>",
                             unsafe_allow_html = True)
            rti_height.write(f"<h1 style='text-align:center; color:red;'>{'{:.1f}'.format(video_width)}</h1>",
                             unsafe_allow_html=True)
            rti_width.write(f"<h1 style='text-align:center; color:red;'>{'{:.1f}'.format(video_height)}</h1>",
                             unsafe_allow_html=True)

            rti_stContainer.image(frame, channels='BGR', use_column_width=True)



            out.write(frame)

        else:
            break

    # release the frame
    st.warning("Infered video saved to : /Users/shubhamrathod/PycharmProjects/nas_streamlit/output")
    out.release()
    cap.release()
    cv2.destroyAllWindows()