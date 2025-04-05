import streamlit as st
from tempfile import NamedTemporaryFile
from utils.image_utils import save_uploaded_file, cleanup_file
from models.traditional import predict_traditional_orb, predict_traditional_sift
from models.deep_learning import predict_baseline_cnn
from models.deep_learning import predict_custom_cct


st.set_page_config(page_title="Pothole Detection")
st.title("🛣️ Comparative Analysis of Traditional and Deep Learning Approaches for Road Pothole Detection")

uploaded_file = st.file_uploader("📸 Upload a road image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file
    temp_path = save_uploaded_file(uploaded_file)
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # --- Traditional Models ---
    trad1_cat, trad1_conf = predict_traditional_orb(temp_path)
    trad2_cat, trad2_conf = predict_traditional_sift(temp_path)

    # --- Deep Learning Models ---
    dl1_cat, dl1_conf = predict_baseline_cnn(uploaded_file)
    dl2_cat, dl2_conf = predict_custom_cct(uploaded_file)

    # --- Display in 2x2 Grid ---
    st.markdown("## 🔍 Model Predictions")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧠 Traditional Model 1 (ORB + SVM)")
        st.markdown(f"**Prediction:** `{trad1_cat.upper()}`")
        st.markdown(f"**Confidence:** `{trad1_conf * 100:.2f}%`")

    with col2:
        st.subheader("🧠 Traditional Model 2 (SIFT + SVM)")
        st.markdown(f"**Prediction:** `{trad2_cat.upper()}`")
        st.markdown(f"**Confidence:** `{trad2_conf * 100:.2f}%`")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("🤖 Baseline CNN Model")
        st.markdown(f"**Prediction:** `{dl1_cat.upper()}`")
        st.markdown(f"**Confidence:** `{dl1_conf * 100:.2f}%`")

    with col4:
        st.subheader("🤖 Custom CCT Model")
        st.markdown(f"**Prediction:** `{dl2_cat.upper()}`")
        st.markdown(f"**Confidence:** `{dl2_conf * 100:.2f}%`")

    # Clean up temp file
    cleanup_file(temp_path)


st.markdown("---")
st.markdown("### Developed With ❤️ By:")

st.markdown("""
| Name                | Roll Number   |
|---------------------|---------------|
| Akansha Gautam      | M23CSA506     |
| Anchit Mulye        | M23CSA507     |
| Om Prakash Solanki  | M23CSA521     |
| Shyam Vyas          | M23CSA545     |
""")
