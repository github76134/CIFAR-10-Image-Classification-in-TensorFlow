
# Tài liệu này được tham khảo tại https://github.com/Sarthak-1408/Cifar-10
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf

class_name = ['máy bay', 'xe ô tô', 'con chim', 'con mèo',
    'con nai', 'con chó', 'con ếch', 'con ngựa', 'tàu thuỷ', 'xe tải',]

model = load_model('my_model.h5')

# Tạo tiêu đề web App
st.title("Phân loại hình ảnh với Bộ dữ liệu Cifar10")
st.header("Vui lòng tải lên hình ảnh liên quan đến điều này ...")
st.text(class_name)

# tạo trình tải ảnh dưới dạng "jpg" , "png", "jpeg"
file = st.file_uploader("Tải hình ảnh lên", type=["jpg", "png", "jpeg"])

# Hàm này sẽ nhận vào hình ảnh và model sau đó dự đoán hình ảnh
def import_and_predict(image_data , model):
    image = ImageOps.fit(image_data , (32 ,32) , Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction

if st.button("Dự đoán"):
    image = Image.open(file)
    st.image(image , use_column_width=True)
    predictions = import_and_predict(image , model)
    string = "Hình ảnh này được dự doán là: " + class_name[np.argmax(predictions)]
    st.success(string)
    
    