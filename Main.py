import cv2
import os
import numpy as np
import joblib
import streamlit as st
from tempfile import NamedTemporaryFile

st.title("🚧 CV Project M23CSA507 - Pothole Detection")

model_choice = st.selectbox("Choose Model Type", ["Traditional (SVM + ORB)", "Deep Learning (CNN)"])

if model_choice == "Traditional (SVM + ORB)":
    kmeans = joblib.load('saved_model/kmeans.pkl')
    svm = joblib.load('saved_model/svm.pkl')
    orb = cv2.ORB_create(nfeatures=1000)
else:
    # cnn_model = load_model('saved_model/cnn_model.h5')
    # TODO
    print("TO BE IMPLEMENTED")
    pass

# Categories
categories = ['normal', 'potholes']


def predict_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is not None:
        words = kmeans.predict(descriptors)
        hist, _ = np.histogram(words, bins=np.arange(kmeans.n_clusters + 1))
    else:
        hist = np.zeros(kmeans.n_clusters)

    prediction = svm.predict([hist])[0]
    prob = svm.predict_proba([hist])[0]

    category = categories[prediction]
    return category, prob[prediction]


# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded image temporarily
    with NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name

    # Display uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Predict
    category, confidence = predict_image(temp_path)

    # Display results
    st.markdown(f"### ➡️ Prediction: **{category.upper()}**")
    st.markdown(f"### 📊 Confidence: **{confidence * 100:.2f}%**")

    # Optional: clean up temp file (optional, Streamlit auto-cleans in temp dirs)
    os.remove(temp_path)

# Footer
st.markdown("---")
st.write("Developed with ❤️ M23CSA507")
