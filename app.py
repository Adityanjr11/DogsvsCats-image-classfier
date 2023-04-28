import streamlit as st
import tensorflow as tf

def load_model():
    model = tf.keras.models.load_model('model.h5')
    return model

model = load_model()

st.write("Dogs vs Cats Classifier")

file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])


from PIL import Image, ImageOps
import numpy as np

def import_and_predict(image_data, model):
    size = (256,256)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img.reshape((1,256,256,3))
    prediction = model.predict(img_reshape)

    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names = ['Dog', 'Cat']
    if predictions == 1:
        string = "This image is most likely a : Dog"
    else:
        string = "This image is most likely a : Cat"

    st.success(string)