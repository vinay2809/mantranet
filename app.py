!mkdir mantra-net-app
!cd mantra-net-app
!touch app.py
import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Step 1: Load the pretrained model
@st.cache_resource
def load_model():
    model_path = 'ManTraNet_Pytorch_Pretrained_Model.pth'
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

# Step 2: Preprocess the uploaded image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Step 3: Generate and display the heatmap
def generate_heatmap(pred_map):
    fig, ax = plt.subplots()
    ax.imshow(pred_map[0], cmap='hot')  # Show first channel only
    ax.axis('off')
    st.pyplot(fig)

# Step 4: Streamlit UI
st.set_page_config(page_title="ManTra-Net Image Forgery Detection", layout="centered")
st.title("🕵️‍♂️ ManTra-Net Forgery Detection")
st.markdown("Upload an image to detect manipulated regions using the pretrained ManTra-Net model.")

uploaded_file = st.file_uploader("📤 Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    model = load_model()
    st.info("Detecting forgery... Please wait.")

    img_tensor = preprocess_image(image)

    with torch.no_grad():
        prediction = model(img_tensor).squeeze(0).numpy()

    st.success("Forgery heatmap generated below:")
    generate_heatmap(prediction)

cd /content/mantra-net-app
!streamlit run app.py

