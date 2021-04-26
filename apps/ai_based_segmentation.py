import cv2
import keras
import streamlit as st
import numpy as np
import pandas as pd
from streamlit import caching
from keras.models import load_model
from PIL import Image
from data.create_data import create_table
import time
from utils.learning.metrics import dice_coef, precision, recall
from utils.io.data import save_results, load_test_images, DataGen
from models.deeplab import Deeplabv3, relu6, BilinearUpsampling, DepthwiseConv2D
from utils.postprocessing.hole_filling import fill_holes
from utils.postprocessing.remove_small_noise import remove_small_areas

input_dim_x = 224
input_dim_y = 224
path = 'segmentation/'
color_space = 'rgb'


@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img


##############
# Model Load #
##############
def wound_image_prediction(bytes_data):
    file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
    input_image = cv2.imdecode(file_bytes, 1)
    cv2.imwrite("segmentation/test/images/1.png", input_image)

    data_gen = DataGen(path, split_ratio=0.0, x=input_dim_x, y=input_dim_y, color_space=color_space)
    x_test, test_label_filenames_list = load_test_images(path)
    keras.backend.clear_session()
    if model_selection == 'UNET':
        model = load_model('{}.hdf5'.format(model_selection),
                           custom_objects={'dice_coef': dice_coef,
                                           'precision': precision,
                                           'recall': recall,
                                           })
        for image_batch, label_batch in data_gen.generate_data(batch_size=len(x_test), test=True):
            prediction = model.predict(image_batch, verbose=1)
            save_results(prediction, 'rgb', path + 'test/predictions/', test_label_filenames_list)
            return prediction

    elif model_selection == "MobilenetV2":
        model = load_model('{}.hdf5'.format(model_selection),
                           custom_objects={'dice_coef': dice_coef,
                                           'precision': precision,
                                           'recall': recall,
                                           'relu6': relu6,
                                           'DepthwiseConv2D': DepthwiseConv2D,
                                           'BilinearUpsampling': BilinearUpsampling
                                           })

        for image_batch, label_batch in data_gen.generate_data(batch_size=len(x_test), test=True):
            prediction = model.predict(image_batch, verbose=1)
            save_results(prediction, 'rgb', path + 'test/predictions/', test_label_filenames_list)
            return prediction
    elif model_selection == "SegNet":
        model = load_model('{}.hdf5'.format(model_selection),
                           custom_objects={'dice_coef': dice_coef,
                                           'precision': precision,
                                           'recall': recall,
                                           })
        for image_batch, label_batch in data_gen.generate_data(batch_size=len(x_test), test=True):
            prediction = model.predict(image_batch, verbose=1)
            save_results(prediction, 'rgb', path + 'test/predictions/', test_label_filenames_list)
            return prediction


def post_process():
    pred_dir = 'segmentation/test/predictions/1.png'
    img = cv2.imread(pred_dir)
    threshold = 50
    _, threshed = cv2.threshold(img, threshold, 255, type=cv2.THRESH_BINARY)
    filled = fill_holes(threshed, threshold, 0.1)
    denoised = remove_small_areas(filled, threshold, 0.05)
    cv2.imwrite('postprocess/' + '1.png', denoised)
    return denoised

###############
# Main Screen #
###############


def app():
    global original_img, bytes_data, model_selection
    st.write('# 🧠Wound Segmentation-Based on Deep Learning')
    st.header('Usage Principle')

    st.markdown("""
        This Page is based on Deep Learning. At present it can choose different models like  UNET, MobilenetV2 SegNet.🤩🤩
        
        ---
        """, unsafe_allow_html=True)

    st.markdown("### Load Image ")
    upload_file = st.file_uploader('💾 Upload a Wound Image', type=['jpg', 'jpeg', 'png'])
    caching.clear_cache()
    st.markdown("### Model Selection ")
    model_selection = st.selectbox(
        "Select a Deep Learning Model",
        ('UNET', 'MobilenetV2', 'SegNet'),
    )
    print(model_selection)
    # model = load_model('{}.hdf5'.format(model_selection)
    # , custom_objects={'dice_coef': dice_coef, 'precision': precision, 'recall': recall, })
    col1, col2 = st.beta_columns(2)
    with col1:
        if upload_file is not None:
            bytes_data = upload_file.read()
            original_img = load_image(upload_file)
            original_img = original_img.resize((256, 256))
            st.write('### Original Image')
            st.image(original_img)

    col2.write('### Button')

    clicked2 = col2.button('Predict Image')
    clicked3 = col2.button('Denoise Prediction')

    if clicked2:
        st.write('### Prediction Image')
        prediction_result = wound_image_prediction(bytes_data)
        # prediction_result_resize = prediction_result.resize((256, 256))
        with st.spinner('⏳Waiting for Prediction...'):
            time.sleep(1)
            st.success('Prediction Complete!')
            st.image(prediction_result, clamp=True)

    if clicked3:
        st.write('### Post-process Image')
        denoise_result = post_process()
        with st.spinner('⏳Waiting for Post-processing...'):
            time.sleep(1)
            st.success('Denoise Complete!')
            st.image(denoise_result, clamp=True)
