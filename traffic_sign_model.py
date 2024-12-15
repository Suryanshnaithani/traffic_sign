import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import os


      
def load_model():
    model = tf.keras.models.load_model('traffic_sign_model.h5')
    return model

def import_and_predict(image_data, model):
        target_size=(150, 150)
        image_data = image_data.resize(target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(image_data)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        return predictions , score

def main():
    st.title("Traffic Sign Recognition")
    st.text("Provide an image of a traffic sign for classification")
    
    model = load_model()
    
    file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
    
    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        predictions = import_and_predict(image, model)
        st.write("This image most likely belongs to class {} with a {:.2f} percent confidence."
             .format(np.argmax(predictions[0]), 100 * np.max(predictions[0])))
        
        


if __name__ == '__main__':
    main()
