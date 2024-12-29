import streamlit as st
import numpy as np
from PIL import Image as PILImage
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.preprocessing import image as keras_image

def initialize_model():
    global model
    model = keras_load_model('model.h5')
    st.write("Model loaded!")

def load_image(uploaded_file):
    img = PILImage.open(uploaded_file)
    img = img.resize((224, 224))
    img = img.convert('RGB')
    img_array = keras_image.img_to_array(img)
    img_tensor = np.expand_dims(img_array, axis=0)
    img_tensor /= 255.0
    print(f"Image shape: {img_tensor.shape}")
    return img_tensor

def prediction(uploaded_file):
    new_image = load_image(uploaded_file)
    pred = model.predict(new_image)
    
    labels = np.array(pred)
    labels[labels >= 0.6] = 1
    labels[labels < 0.6] = 0
    
    final = np.array(labels)
    
    if final[0][0] == 1:
        return "Bad"
    else:
        return "Good"

# Streamlit UI
st.title('Image Classifier')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")

    if st.button('Classify'):
        initialize_model()  # Load the model
        prediction_result = prediction(uploaded_file)
        st.write(f"Prediction: {prediction_result}")
