import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as T
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import io

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="VisionExtract",
    page_icon="🖼️",
    layout="wide"
)

# -------------------------------
# Custom UI Styling
# -------------------------------
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #4CAF50;
    }
    .sub-text {
        text-align: center;
        font-size: 18px;
        color: gray;
        margin-bottom: 30px;
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        padding: 30px;
        border-radius: 10px;
        text-align: center;
        background-color: #f9f9f9;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Title Section
# -------------------------------
st.markdown('<p class="main-title">VisionExtract</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Subject Isolation from Images</p>', unsafe_allow_html=True)

# -------------------------------
# Load Model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
)

model_path = "best_model.pth"
if not os.path.exists(model_path):
    st.error("Model file not found. Please train the model first.")
    st.stop()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
# -------------------------------
# Upload Section
# -------------------------------
st.markdown("### 📤 Upload Image")

st.markdown('<div class="upload-box">Drag & Drop Image Here 👇</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["png", "jpg", "jpeg","webp"],
    label_visibility="collapsed"
)

# -------------------------------
# Processing
# -------------------------------
if uploaded_file is not None:
    with st.spinner("Processing Image... Please wait ⏳"):

        image = Image.open(uploaded_file).convert("RGB")
        orig_width, orig_height = image.size

        # Match the training dataset's RGB resize and normalization.
        image_array = np.asarray(image)
        image_array = cv2.resize(image_array, (256, 256))
        image_array = image_array.astype("float32") / 255.0
        input_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).to(device)

        # Model inference
        with torch.no_grad():
            mask_pred = model(input_tensor)
            mask_pred = torch.sigmoid(mask_pred)

        # Resize mask back to original size
        mask_pred = F.interpolate(
            mask_pred,
            size=(orig_height, orig_width),
            mode='bilinear',
            align_corners=False
        )

        mask_pred = (mask_pred.squeeze().cpu() > 0.5).float()

        # Apply mask
        original_tensor = T.ToTensor()(image)
        isolated_tensor = original_tensor * mask_pred.unsqueeze(0)
        isolated_image = T.ToPILImage()(isolated_tensor)

    # -------------------------------
    # Display Results
    # -------------------------------
    st.markdown("### 🖼️ Result Preview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Original Image")
        st.image(image, width=450)

    with col2:
        st.markdown("#### Isolated Subject")
        st.image(isolated_image, width=450)

    # -------------------------------
    # Download Button
    # -------------------------------
    buf = io.BytesIO()
    isolated_image.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.markdown("### 📥 Download Output")

    st.download_button(
        label="⬇️ Download Isolated Image",
        data=byte_im,
        file_name="isolated_image.png",
        mime="image/png"
    )