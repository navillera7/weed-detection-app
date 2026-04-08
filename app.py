import streamlit as st
from ultralytics import YOLO
from PIL import Image
st.title("🌱 AI Weed Detection")
st.write("Upload your plant images, and let the AI detect weeds for you! This tool uses a YOLO model trained to identify weeds in your photos.")


@st.cache_resource
def load_model():
    return YOLO('best.pt') 

model = load_model()

uploaded_file = st.file_uploader("Press here to upload the image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Image', use_container_width=True)
    if st.button('Run'):
        with st.spinner('AI is analyzing the image...'):
            results = model(image)
            
            res_image = results[0].plot() 

            st.success("Done!")
            st.image(res_image, caption='Result:', use_container_width=True)