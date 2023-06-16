import numpy as np
import streamlit as st
import os
from PIL import Image
from object_detection_image_video_streamlit import *
import tempfile

# hide hamburger and customize footer
hide_menu= """
<style>

#MainMenu {
    visibility:hidden;
}

footer{
    visibility:visible;
}

footer:after{
    content: 'With 🫶️ from Shubham Shankar.';
    display:block;
    position:relative;
    color:grey;
    padding;5px;
    top:3px;
}
</style>

"""


def main():
    st.image("/Users/shubhamrathod/PycharmProjects/nas_streamlit/icon_image/icon.png", width=85)
    st.title("YOLO-NAS Object Detection")
    st.markdown(hide_menu, unsafe_allow_html=True)

    # save image file -----------------------------------------------------------------------
    def save_uploaded_image_file(uploadedfile):
        with open(os.path.join("/Users/shubhamrathod/PycharmProjects/nas_streamlit/input/images/", uploadedfile.name), "wb") as f:
            f.write(uploadedfile.getbuffer())
        return st.success("Input file Saved in location:{} to /Users/shubhamrathod/PycharmProjects/nas_streamlit/input/images".format(uploadedfile.name))

    # save video file
    def save_uploaded_video_file(uploadedfile):
        with open(
                os.path.join("/Users/shubhamrathod/PycharmProjects/nas_streamlit/input/video/", uploadedfile.name),
                "wb") as f:
            f.write(uploadedfile.getbuffer())
        return st.success("Saved File:{} to /Users/shubhamrathod/PycharmProjects/nas_streamlit/input/video".format(
            uploadedfile.name))

    # save video file
    def save_uploaded_rti_video_file(uploadedfile):
        with open(
                os.path.join("/Users/shubhamrathod/PycharmProjects/nas_streamlit/input/video/", uploadedfile.name),
                "wb") as f:
            f.write(uploadedfile.getbuffer())
        return st.success("Saved File:{} to /Users/shubhamrathod/PycharmProjects/nas_streamlit/input/video".format(
            uploadedfile.name))

    # --------------------------------------------------------------------------------------------------------------------------------------------------------

    st.sidebar.title("Inference")
    st.sidebar.markdown('---')

    # Using object notation
    add_model = st.sidebar.selectbox(
        "Select Task to be Performed",
        ['Home Page', 'Image Inference', 'Video Inference', 'Real Time Inference']
    )

    if add_model == 'Home Page':

        st.write(
            """
            
            Hi 👋, I'm **:red[Shubham Shankar]**, and welcome to my **:green[Object Detection Application]**! :rocket: This program makes use of the **:blue[YOLO-NAS]** model, 
            which was specially trained using the **:violet[Roboflow]** mask detection dataset.  ✨
            
            """
        )

        st.markdown('---')

        st.write(
            """
            ### App Interface!!
            
            :dog: The web app has an easy-to-use interface. 
            
            Select the task you wish to carry out from the drop-down menu on the left side:
            
            1] **:green[Image Inference]**: Upload an image using the provided button. The app will perform inference on the image using a  machine learning model and display the results.
            
            2] **:violet[Video Inference]**: Upload a video file using the provided button. The app will perform inference on the video frames using a  machine learning model and display the results.
            
            3] **:red[Real Time Inference]**: Click the "Start Webcam" button to activate your webcam. The app will perform inference on the webcam video feed using a  machine learning model and display the results.

            """
        )

        st.markdown('---')

        st.info(
            """
            Visit this page to learn more about [YOLO-NAS](https://deci.ai/blog/yolo-nas-object-detection-foundation-model/) and to explore [Robolfow](https://roboflow.com/).
            """,
            icon="👾",
        )

        st.markdown('---')

        st.image('/Users/shubhamrathod/PycharmProjects/nas_streamlit/icon_image/pipeline.png')

        st.markdown('---')


        st.error(
            """
            Connect with me on [**Github**](https://github.com/RATHOD-SHUBHAM) and [**LinkedIn**](https://www.linkedin.com/in/shubhamshankar/). ✨
            """,
            icon="🧟‍♂️",
        )

        st.markdown('---')

    elif add_model == 'Image Inference':
        st.sidebar.markdown('---')
        # conf = st.sidebar.slider('Select Confidence Value', min_value= 0.0, max_value=1.0, value=0.50)
        # st.sidebar.markdown('---')
        conf = st.slider('Select Confidence Value', min_value=0.0, max_value=1.0, value=0.50)


        # img_file = st.sidebar.file_uploader("Upload an Image", type=['png', 'jpeg', 'jpg'])
        img_file = st.file_uploader("Upload an Image", type=['png', 'jpeg', 'jpg'])
        if img_file is not None:

            if st.button('Infer Image'):
                st.markdown('---')
                save_uploaded_image_file(img_file)
                st.markdown('---')

                file_details = {"FileName": img_file.name, "FileType": img_file.type}
                st.success(file_details)
                st.markdown('---')

                # https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html
                img = cv2.imdecode(np.fromstring(img_file.read(), np.uint8) , 1)
                image = np.array(Image.open(img_file))

                # st.sidebar.text("Raw Image")
                # st.sidebar.image(image)
                # st.text("Raw Image")
                # st.image(image)

                load_yolonas_process_image(img, conf)

    elif add_model == 'Video Inference':
        # v_conf = st.sidebar.slider('Select Confidence Value', min_value=0.0, max_value=1.0, value=0.75)
        # st.sidebar.markdown('---')
        v_conf = st.slider('Select Confidence Value', min_value=0.0, max_value=1.0, value=0.75)
        st.sidebar.markdown('---')


        temporary_folder = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)

        # video_file = st.sidebar.file_uploader("Upload an Image", type=['mp4', 'webm', "avi", "mov"])
        video_file = st.file_uploader("Upload an Image", type=['mp4', 'webm', "avi", "mov"])
        if video_file is not None:
            if st.button('Infer Video'):
                save_uploaded_video_file(video_file)
                st.markdown('---')

                video_file_details = {"FileName": video_file.name, "FileType": video_file.type}
                st.success(video_file_details)

                temporary_folder.write(video_file.read())
                temp_video = open(temporary_folder.name, 'rb')
                temp_video_bytes = temp_video.read()
                st.sidebar.text('Input Video')
                st.sidebar.video(temp_video_bytes)
                st.markdown("<hr/>", unsafe_allow_html=True)
                col_1, col_2, col_3 = st.columns(3)
                with col_1:
                    st.markdown("**Frame Rate**")
                    v_frame_rate_text = st.markdown("0")
                with col_2:
                    st.markdown("**Video Width**")
                    v_width_text = st.markdown("0")
                with col_3:
                    st.markdown("**Video Height**")
                    v_height_text = st.markdown("0")
                st.markdown("<hr/>", unsafe_allow_html=True)

                stContainer = st.empty()

                load_yolonas_process_video(temporary_folder.name, v_conf, v_frame_rate_text, v_height_text, v_width_text, stContainer)

    elif add_model == 'Real Time Inference':
        rti_conf = st.slider('Select Confidence Value', min_value=0.0, max_value=1.0, value=0.75)
        st.markdown('---')


        if st.button('Start Webcam'):

            st.markdown("<hr/>", unsafe_allow_html=True)

            rti_col_1, rti_col_2, rti_col_3 = st.columns(3)
            with rti_col_1:
                st.markdown("**Frame Rate**")
                rti_frame_rate = st.markdown("0")
            with rti_col_2:
                st.markdown("**Video Width**")
                rti_width = st.markdown("0")
            with rti_col_3:
                st.markdown("**Video Height**")
                rti_height = st.markdown("0")
            st.markdown("<hr/>", unsafe_allow_html=True)

            rti_stContainer = st.empty()

            load_yolonas_process_RTI_video(rti_conf, rti_frame_rate, rti_height, rti_width, rti_stContainer)


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass

