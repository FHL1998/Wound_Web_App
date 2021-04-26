import streamlit as st
import pandas as pd
import numpy as np
from data.create_data import create_table

import time

from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage import morphology
import cv2
from streamlit import caching
from streamlit.components.v1 import components
from streamlit_cropper import st_cropper
from PIL import Image


def wound_segmentation(bytes_data, roi, res):
    # Image manipulation

    file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
    input_image = cv2.imdecode(file_bytes, 1)  # cv2.imdecode()函数从指定的内存缓存中读取数据，并把数据转换(解码)成图像格式;主要用于从网络传输数据中恢复出图像。

    height = input_image.shape[0]  # image.shape[0]#图片垂直尺寸
    width = input_image.shape[1]  # image.shape[1]#图片水平尺寸
    channel = input_image.shape[2]  # 图片通道数
    ratio = height / width

    # ----------------------------------------------------------
    # Resize the image while keeping the same height-width ratio
    # ----------------------------------------------------------

    if width != 256:
        width = 256
        height = int(width * ratio)
    input_image = cv2.resize(input_image, (width, height))
    input_image_RGB = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # BGR to RGB
    input_image_Gray = cv2.cvtColor(input_image_RGB, cv2.COLOR_RGB2GRAY)  # RGB to Gray
    input_image_Lab = cv2.cvtColor(input_image_RGB, cv2.COLOR_RGB2LAB)  # RGB to LAB
    l, a, b = cv2.split(input_image_Lab)  # 图像通道的拆分
    # L表示照度（Luminosity）,相当于亮度
    # a表示从洋红色至绿色的范围
    # b表示从黄色至蓝色的范围

    x1 = int(roi[0])
    x2 = int(roi[1])
    y1 = int(roi[2])
    y2 = int(roi[3])

    mask = np.zeros(input_image_Gray.shape, dtype=bool)
    mask[y1:y2, x1:x2] = True

    # -------------------------------------------
    # Gaussian Blur and Edge detection(Canny Edge)
    # -------------------------------------------

    low_threshold = st.slider("Threshold:", 1, 100, 30)  # low_threshold = 30
    ratio = st.slider("Ratio:", 2.0, 4.0, 3.0)
    threshold2 = low_threshold * ratio
    kernel_size = 3
    image_blur = cv2.GaussianBlur(input_image_Gray, (3, 3), 0.8, 0.8)
    image_edge = cv2.Canny(image_blur, low_threshold, threshold2, kernel_size)
    image_edge[mask == 0] = 0
    # Ie[mask == 0] = 0

    # ---------------------------------
    # Morphological operations 形态运算
    # ---------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # kernel 代表腐蚀操作时所采用的的结构类型。
    # cv2.getStructuringElement( ) 返回指定形状和尺寸的结构元素
    image_closing = cv2.morphologyEx(image_edge, cv2.MORPH_CLOSE, kernel,
                                     iterations=6)  # 闭运算是先膨胀，后腐蚀的运算，它有助于关闭前景物体内部的小孔，或去除物体上的小黑点，还可以将不同的前景图像进行连接
    image_closing_filling_holes = ndi.binary_fill_holes(image_closing)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    image_erode_boolean = cv2.erode(np.float32(image_closing_filling_holes), erode_kernel, iterations=3).astype(
        bool)  # astype：转换数组的数据类型

    # ---------------------------------
    # Label segmented regions
    # ---------------------------------
    label_img = label(image_erode_boolean)
    regions = regionprops(label_img)

    areas = []
    for i in range(len(regions)):
        areas.append(regions[i].area)

    max_area = np.max(areas)
    # print(max_area)
    # print(image_erode_boolean)
    image_remove_small_object = morphology.remove_small_objects(image_erode_boolean, max_area - 1)
    # image_erode_boolean待操作的bool型数组
    # (max_area - 1)表示最小连通区域尺寸，小于该尺寸的都将被删除

    image_remove_small_object = np.uint8(image_remove_small_object)
    image_remove_small_object[image_remove_small_object == 1] = 255
    # convert back to 8 bit values
    input_image_RGB_copy = np.uint8(input_image_RGB.copy())  # 预先复制一份，将该副本图像传递给函数cv2.drawContours()
    img = image_remove_small_object
    return_threshold, segmentation_image = cv2.threshold(img, 127, 255,
                                                         cv2.THRESH_BINARY)  # thresh(设定的阈值)为127，maxval 代表参数为THRESH_BINARY(二值化阈值处理） 或 THRESH_BINARY_INV时，需要设定的最大值
    contours, hierarchy = cv2.findContours(segmentation_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = cv2.drawContours(input_image_RGB_copy, contours, 0, (0, 255, 0), 2)  # 轮廓的颜色为(0,255,0),thickness 为2

    # ------------------------
    # Feature Calculations
    # ------------------------
    props = regions[0]

    estimated_Area = props["area"]
    estimated_Perimeter = props["perimeter"]
    major_length_Axis = props["major_axis_length"]
    minor_length_Axis = props["minor_axis_length"]

    Area = round(estimated_Area * res ** 2, 2)
    ThreeD_Area = round(1.25 * Area, 2)
    Perimeter = round(estimated_Perimeter * res, 2)
    major_length_Axis_round = round(major_length_Axis * res, 2)
    minor_length_Axis_round = round(minor_length_Axis * res, 2)

    # --------------------------------
    # Tissue Typing
    # --------------------------------
    fI = np.uint8(input_image_Lab.copy())
    Il = cv2.cvtColor(input_image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(Il)

    fg = cv2.bitwise_and(fI, fI, mask=image_remove_small_object)
    Is = cv2.bitwise_and(input_image, input_image, mask=image_remove_small_object)
    fgc = fg[:, :, [1, 2]]
    im_reshaped = fgc.reshape(fgc.shape[0] * fgc.shape[1], fgc.shape[2])
    im_reshaped = np.float32(im_reshaped)
    #    im_reshaped[im_reshaped==0]=float("NaN")
    #    im_reshaped = fg.reshape((-1, 3))
    im_reshaped = np.float32(im_reshaped)
    # define criteria and apply kmeans()
    k = 4
    ref = np.array([[165, 160], [135, 150], [120, 120], [0, 0]])
    #    ref = np.array([[165,160],[135,150],[127,127],[0,0]])
    #    kmeans = KMeans(n_clusters=k, init=seed).fit(im_reshaped)
    #    centers = kmeans.cluster_centers_
    #    labels = kmeans.labels_

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, labels, centers = cv2.kmeans(im_reshaped, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert back to 8 bit values
    centers = np.uint8(centers)
    #
    #    # flatten the labels array
    labels = labels.flatten()
    #    st.write("Seed: ",seed,"centers: ", centers, "Labels: ", labels)

    dist = np.zeros(k)
    tissue_class = np.zeros(k)

    for i in range(k):
        for j in range(k):
            dist[j - 1] = np.linalg.norm(centers[i - 1] - ref[j - 1])
        tissue_class[i - 1] = np.argmin(dist)

    g = np.zeros(fg.shape, dtype='uint8')
    s = np.zeros(fg.shape, dtype='uint8')
    n = np.zeros(fg.shape, dtype='uint8')

    g = g.reshape((-1, 3))
    s = s.reshape((-1, 3))
    n = n.reshape((-1, 3))

    for i in range(k):
        if tissue_class[i] == 0:
            g[labels == i] = [1, 1, 1]

        if tissue_class[i] == 1:
            s[labels == i] = [1, 1, 1]

        if tissue_class[i] == 2:
            n[labels == i] = [1, 1, 1]

    g = g.reshape(fg.shape)
    s = s.reshape(fg.shape)
    n = n.reshape(fg.shape)

    gPixels = g.sum()
    sPixels = s.sum()
    nPixels = n.sum()

    totP = gPixels + sPixels + nPixels
    gP = round(gPixels * 100 / totP, 1)
    sP = round(sPixels * 100 / totP, 1)
    nP = round(nPixels * 100 / totP, 1)

    gI = cv2.bitwise_and(Is, input_image, mask=g[:, :, 1])
    sI = cv2.bitwise_and(Is, input_image, mask=s[:, :, 1])
    nI = cv2.bitwise_and(Is, input_image, mask=n[:, :, 1])

    bg = np.zeros(fg.shape, dtype='uint8')
    bg[:, :] = (2, 89, 15)

    bw4 = cv2.cvtColor(gI, cv2.COLOR_RGB2GRAY)
    bw4[bw4 != 0] = 255
    bw4 = cv2.bitwise_not(bw4)
    bgt = cv2.bitwise_and(bg, bg, mask=bw4)
    gI = cv2.add(gI, bgt)

    bw4 = cv2.cvtColor(sI, cv2.COLOR_RGB2GRAY)
    bw4[bw4 != 0] = 255
    bw4 = cv2.bitwise_not(bw4)
    bgt = cv2.bitwise_and(bg, bg, mask=bw4)
    sI = cv2.add(sI, bgt)

    bw4 = cv2.cvtColor(nI, cv2.COLOR_RGB2GRAY)
    bw4[bw4 != 0] = 255
    bw4 = cv2.bitwise_not(bw4)
    bgt = cv2.bitwise_and(bg, bg, mask=bw4)
    nI = cv2.add(nI, bgt)
    # Cluster 0 Mask
    mask0 = np.zeros(fg.shape, dtype='uint8')
    mask0 = mask0.reshape((-1, 3))
    cluster = 0
    mask0[labels == cluster] = [1, 1, 1]
    mask0 = mask0.reshape(fg.shape)

    # Cluster 1 Mask
    mask1 = np.zeros(fg.shape, dtype='uint8')
    mask1 = mask1.reshape((-1, 3))
    cluster = 1
    mask1[labels == cluster] = [1, 1, 1]
    mask1 = mask1.reshape(fg.shape)

    # Cluster 2 Mask
    mask2 = np.zeros(fg.shape, dtype='uint8')
    mask2 = mask2.reshape((-1, 3))
    cluster = 2
    mask2[labels == cluster] = [1, 1, 1]
    mask2 = mask2.reshape(fg.shape)

    #    #Cluster 3 Mask
    #    mask3 = np.zeros(fg.shape,dtype='uint8')
    #    mask3 = mask3.reshape((-1,3))
    #    cluster = 3
    #    mask3[labels==cluster] = [1,1,1]
    #    mask3 = mask3.reshape(fg.shape)

    # Apply Masks
    t1 = cv2.bitwise_and(Is, input_image, mask=mask0[:, :, 1])
    t2 = cv2.bitwise_and(Is, input_image, mask=mask1[:, :, 1])
    t3 = cv2.bitwise_and(Is, input_image, mask=mask2[:, :, 1])
    #    t4 = cv2.bitwise_and(Is, I, mask=mask3[:,:,1])

    return contour, Area, ThreeD_Area, Perimeter, major_length_Axis_round, minor_length_Axis_round, gI, sI, nI, gP, sP, nP, t1, t2, t3


@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img


def app():
    global saxis, laxis, perim, area, threed_area, fig, sP, gP, nP, gI, sI, nI
    st.write('# 🧠Wound Segmentation-Based on OpenCV')
    st.header('Usage Principle')
    st.markdown("""
    **Step-1**. Calibrate image.The calibration of the image can be realized by dragging the corners of the marker over two points of known distance in the horizontal axis and enter the distance.

    **Step-2**. Adjust the bounding box to select the wound(Region Of Interest).

    **Step-3**. Adjust the Threshold and Width sliders to get the ideal segmentation.

    Download sample wound images from [here](https://github.com/TOESL100/demoimages).

    ---
    """, unsafe_allow_html=True)

    st.markdown("### Load Image ")
    img_file = st.file_uploader(label='💾 Upload a Wound Image', type=['png', 'jpg', 'jpeg'])
    caching.clear_cache()

    if img_file is not None:
        bytes_data = img_file.read()
        img = load_image(img_file)
        #    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        #    I = cv2.imdecode(file_bytes, 1)
        # Resize Image
        w, h = img.size
        r = h / w

        if w != 256:
            w = 256
            h = int(w * r)
        img = img.resize((w, h))

        #    st.image(I)

        # Calibrate Image
        st.markdown(
            """To calibrate image place top left and right edges of the box over two points of known distance in 
        millimeters.""")
        box = st_cropper(img, realtime_update=True,
                         aspect_ratio=(1, 0.01),
                         box_color="green",
                         return_type='box')
        dist = st.number_input("Distance in mm:", value=30)
        w = box["width"]
        res = dist / w
        st.write("Image Resolution = ", round(res, 2), "  mm/pixel")

        #    # ROI
        st.markdown("### Select ROI")
        box = st_cropper(img, realtime_update=True,
                         aspect_ratio=None,
                         box_color="green",
                         return_type='box', key=2)

        roi = np.zeros(4)
        roi[0] = box["left"]
        roi[1] = box["left"] + box["width"] - 1
        roi[2] = box["top"]
        roi[3] = box["top"] + box["height"] - 1

        #    #Segment Image
        if bytes_data is not None:
            fig, area, threed_area, perim, laxis, saxis, gI, sI, nI, gP, sP, nP, t1, t2, t3 = wound_segmentation(
                bytes_data, roi, res)

        st.markdown("### Segmented Wound")
        with st.spinner('⏳ Waiting for Analyzing...'):
            time.sleep(1)
            st.success('Segmentation Complete!')

            st.image(fig)

        st.write("Area: ", area, "square mm")
        st.write("Perimeter: ", perim, "mm")
        st.write("Covering material size required for 3D printing:", threed_area, "square mm")
        st.write("Long Axis: ", laxis, "mm")
        st.write("Short Axis: ", saxis, "mm")

        gText = "Granulation Tissue (" + str(gP) + "%)"  # 肉芽组织
        sText = "Slough (" + str(sP) + "%)"  # 腐肉
        #nText = "Necrotic Tissue (" + str(nP) + "%)"  # 坏死组织

        st.image(gI, caption=gText)
        st.image(sI, caption=sText)
        #st.image(nI, caption=nText)
