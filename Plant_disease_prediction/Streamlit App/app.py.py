import streamlit as st
import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# ----------------------------
# 1) Load Model & Class Indices
# ----------------------------
MODEL_PATH = "plant_vgg16_best.h5"
CLASS_JSON = "class_indices.json"

model = load_model(MODEL_PATH)

with open(CLASS_JSON, "r") as f:
    class_indices = json.load(f)

# Reverse mapping: index -> class name
idx2class = {v:k for k,v in class_indices.items()}

# ----------------------------
# 2) Streamlit UI
# ----------------------------
st.title("🌱 Plant Disease Prediction")
st.write("Upload a leaf image and the model will predict its disease type.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    # ----------------------------
    # 3) Preprocess image
    # ----------------------------
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # ----------------------------
    # 4) Make Prediction
    # ----------------------------
    preds = model.predict(img_array)[0]
    pred_class_idx = np.argmax(preds)
    pred_class_name = idx2class[pred_class_idx]
    confidence = preds[pred_class_idx] * 100

    st.subheader(f"✅ Predicted Class: {pred_class_name}")
    st.write(f"🔹 Confidence: {confidence:.2f}%")

    # ----------------------------
    # 5) Show top 5 predictions
    # ----------------------------
    top_k = 5
    top_indices = preds.argsort()[::-1][:top_k]
    top_classes = [idx2class[i] for i in top_indices]
    top_probs = [preds[i]*100 for i in top_indices]

    st.write("### Top Predictions")
    for c, p in zip(top_classes, top_probs):
        st.write(f"{c}: {p:.2f}%")

    # Bar chart
    plt.figure(figsize=(6,4))
    plt.barh(top_classes[::-1], top_probs[::-1], color='green')
    plt.xlabel("Confidence (%)")
    plt.title("Top 5 Predictions")
    st.pyplot(plt)
