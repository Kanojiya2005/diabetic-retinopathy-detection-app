import streamlit as st
from PIL import Image
from utils import load_model, predict_image

st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    layout="centered"
)

st.title("👁️ Diabetic Retinopathy Detection")
st.write("Upload a retinal fundus image to detect severity level.")

# Load model (cached)
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

uploaded_file = st.file_uploader(
    "Upload Retinal Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    result = predict_image(image, model)

    # Diagnosis Result
    st.subheader("Diagnosis Result")

    severity = result["class_name"]

    style_map = {
        "No DR": st.success,
        "Mild": st.info,
        "Moderate": st.warning,
        "Severe": st.error,
        "Proliferative (Blindness Risk)": st.error
    }

    style_map.get(severity, st.write)(severity)

    # Suggestion
    st.subheader("Medical Suggestion")
    st.success(result["suggestion"])