import streamlit as st
from models import traditional, deep_learning
from utils.image_utils import save_uploaded_file, cleanup_file

st.title("🚧 CV Project M23CSA507 - Pothole Detection")

# Model selection
model_choice = st.selectbox("Choose Model Type", ["Traditional (SVM + ORB)", "Deep Learning (CNN)"])

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded image temporarily
    temp_path = save_uploaded_file(uploaded_file)

    # Display uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Prediction
    if model_choice == "Traditional (SVM + ORB)":
        category, confidence = traditional.predict(temp_path)
    else:
        category, confidence = deep_learning.predict(temp_path)

    # Display results
    st.markdown(f"### ➡️ Prediction: **{category.upper()}**")
    st.markdown(f"### 📊 Confidence: **{confidence * 100:.2f}%**")

    # Clean up temp file
    cleanup_file(temp_path)

# Footer
st.markdown("---")
st.write("Developed with ❤️ M23CSA507")
