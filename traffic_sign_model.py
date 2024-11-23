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
        class_names = [
    "Speed limit (5km/h)",
    "Speed limit (15km/h)",
    "Speed limit (30km/h)",
    "Speed limit (40km/h)",
    "Speed limit (50km/h)",
    "Speed limit (60km/h)",
    "Speed limit (70km/h)",
    "Speed limit (80km/h)",
    "Don't Go straight or left",
    "Don't Go straight or right",
    "Don't Go straight",
    "Don't Go left",
    "Don't Go left or right",
    "Don't Go right",
    "Don't overtake from left",
    "No U-turn",
    "No Car",
    "No horn",
    "Speed limit (40km/h)",
    "Speed limit (50km/h)",
    "Go straight or right",
    "Go straight",
    "Go left",
    "Go left or right",
    "Go right",
    "Keep left",
    "Keep right",
    "Roundabout mandatory",
    "Watch out for cars",
    "Horn",
    "Bicycles crossing",
    "U-turn",
    "Road Divider",
    "Traffic signals",
    "Danger Ahead",
    "Zebra Crossing",
    "Bicycles crossing",
    "Children crossing",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Unknown1",
    "Unknown2",
    "Unknown3",
    "Go right or straight",
    "Go left or straight",
    "Unknown4",
    "ZigZag Curve",
    "Train Crossing",
    "Under Construction",
    "Unknown5",
    "Fences",
    "Heavy Vehicle Accidents",
    "Unknown6",
    "Give Way",
    "No stopping",
    "No entry",
    "Unknown7",
    "Unknown8"
]
        st.write("This image most likely belongs to {} with a {:.2f} percent confidence."
                 .format(class_names[np.argmax(predictions[0])], 100 * np.max(predictions[0])))
        
        


if __name__ == '__main__':
    main()
