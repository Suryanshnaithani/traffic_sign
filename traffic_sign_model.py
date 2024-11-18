import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import os

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('traffic_sign_model.h5')
    return model

def import_and_predict(image_data, model):
        size = (150,150)
        image = Image.open(image_data)
        image = image.resize(size)
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array/255
        prediction = model.predict(img_array)
        return prediction

model = load_model()

st.write("""
            # Traffic Sign Recognition
            """
            )

st.write("This is a simple image classification web app to predict traffic signs")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(file, model)
    score = tf.nn.softmax(predictions[0])
    st.write("This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))
    st.write("The model is trained on the following classes: ", class_names)

## running the app
if __name__ == '__main__':
    main()
