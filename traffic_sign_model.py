import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image

def main():
    st.set_page_config(page_title="Traffic Sign Recognition", page_icon="🚦", layout="centered", initial_sidebar_state="expanded")
    st.title("Traffic Sign Recognition")
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

def import_and_predict(image_data, model):
    size = (30, 30)
    image = Image.open(image_data)
    image = image.resize(size)
    image = np.asarray(image)
    image = image.flatten()  # Flatten the image to match the model's input shape
    image = image.reshape(1, -1)  # Reshape to (1, 25920)
    predictions = model.predict(image)
    return predictions

model = tf.keras.models.load_model("traffic_sign_model.h5")
class_names = ['Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)', 'No passing', 'No passing for vehicles over 3.5 metric tons', 'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right', 'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing', 'End of no passing by vehicles over 3.5 metric tons']

if __name__ == "__main__":
    main()
