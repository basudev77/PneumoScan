import streamlit as st
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model("cnn_model.keras")

st.title("🩻 Chest X-Ray Classification")
st.write("Upload an X-ray image to check if it’s **Normal** or **Pneumonia**.")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded X-ray", use_container_width=True)

    # Preprocess image
    img = img.resize((120, 120))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    prediction = model.predict(img_array)
    prob = prediction[0][0]

    if prob >= 0.5:
        st.error(f"🫁 Pneumonia detected ({prob*100:.2f}%)")
    else:
        st.success(f"✅ Normal ({(1-prob)*100:.2f}%)")
