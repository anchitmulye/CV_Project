import streamlit as st
from tempfile import NamedTemporaryFile
from utils.image_utils import save_uploaded_file, cleanup_file
from models.traditional import predict_traditional
from models.deep_learning import predict_deep_learning

# Helper functions
def save_uploaded_file(uploaded_file):
    with NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return temp_file.name

def cleanup_file(path):
    try:
        os.remove(path)
    except Exception:
        pass

st.set_page_config(page_title="Pothole Detection")
st.title("🛣️ Comparative Analysis of Traditional and Deep Learning Approaches for Road Pothole Detection")

uploaded_file = st.file_uploader("📸 Upload a road image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save and display uploaded image
    temp_path = save_uploaded_file(uploaded_file)
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Run both models
    trad_category, trad_confidence = predict_traditional(temp_path)
    dl_category, dl_confidence = predict_deep_learning(uploaded_file)

    # Structured output
    st.markdown("## 🔍 Model Predictions")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 Traditional Model (SVM + ORB)")
        st.markdown(f"**Prediction:** `{trad_category.upper()}`")
        st.markdown(f"**Confidence:** `{trad_confidence * 100:.2f}%`")

    with col2:
        st.subheader("🤖 Deep Learning Model (CNN)")
        st.markdown(f"**Prediction:** `{dl_category.upper()}`")
        st.markdown(f"**Confidence:** `{dl_confidence * 100:.2f}%`")

    # Clean up
    cleanup_file(temp_path)

st.markdown("---")
st.write("Developed with ❤️ by Shyam Vyas (M23CSA545), Akansha Gautam (M23CSA506), Anchit Mulye (M23CSA507), and Om Prakash Solanki (M23CSA521)")

