import streamlit as st
import os
from tempfile import NamedTemporaryFile

from models.traditional import predict_traditional
from models.deep_learning import predict_deep_learning

st.set_page_config(page_title="Pothole Detector", layout="centered")
st.title("🚧 CV Project M23CSA507 - Pothole Detection")

# Model Selection
model_choice = st.selectbox("Choose Model Type", ["Traditional (SVM + ORB)", "Deep Learning (CNN)"])

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    with NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name

    try:
        if model_choice == "Traditional (SVM + ORB)":
            category, confidence = predict_traditional(temp_path)
        else:
            category, confidence = predict_deep_learning(uploaded_file)

        st.markdown(f"### ➡️ Prediction: **{category.upper()}**")
        st.markdown(f"### 📊 Confidence: **{confidence * 100:.2f}%**")

    except Exception as e:
        st.error(f"Error: {e}")

    finally:
        os.remove(temp_path)

st.markdown("---")
st.write("Developed with ❤️ M23CSA507, M23CSA506")

