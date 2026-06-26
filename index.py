import streamlit as st

st.set_page_config(
    page_title="Oral Cancer AI",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Oral Cancer AI Detection")
st.write("Deep Learning Based Oral Cancer Detection System")

uploaded_file = st.file_uploader(
    "Upload Oral Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image")

    if st.button("🔍 Analyze Image"):
        # Call your TensorFlow model
        prediction = "Cancer"
        confidence = 96.7
        risk = "High"

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Prediction", prediction)

        with col2:
            st.metric("Confidence", f"{confidence:.2f}%")

        with col3:
            st.metric("Risk", risk)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔥 Grad-CAM")
            # st.image(gradcam_image)

        with col2:
            st.subheader("🎯 Score-CAM")
            # st.image(scorecam_image)