import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="Aesthetix AI",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Premium White/Clean Theme
st.markdown("""
    <style>
    /* App Background */
    .stApp {
        background-color: #F8F9FB;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main Content Card Style */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Custom Headers */
    h1 {
        color: #1A1A1A;
        font-weight: 700;
        letter-spacing: -1px;
        text-align: center;
        padding-bottom: 10px;
    }
    
    p {
        color: #666666;
    }
    
    /* Styled Image Containers */
    div[data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 10px 20px rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
    }
    
    /* Score Card */
    .score-card {
        background-color: #FFFFFF;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        text-align: center;
        border: 1px solid #EEEEEE;
        margin-top: 20px;
    }
    
    .score-value {
        font-size: 5rem;
        font-weight: 800;
        margin: 0;
        line-height: 1;
    }
    
    .score-label {
        font-size: 1.1rem;
        color: #888;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(90deg, #1A1A1A 0%, #333333 100%);
        color: white;
        border: none;
        padding: 12px 28px;
        border-radius: 50px;
        font-weight: 600;
        letter-spacing: 0.5px;
        width: 100%;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        background: #000000;
    }
    
    /* File Uploader */
    .stFileUploader {
        padding: 20px;
        background-color: #FFFFFF;
        border-radius: 15px;
        border: 1px dashed #DDDDDD;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1>✨ Aesthetix AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; margin-top: -15px; margin-bottom: 30px;'>Facial Symmetry & Feature Analysis Engine</p>", unsafe_allow_html=True)

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_models():
    device = torch.device("cpu")
    
    # Rating Model (ResNet18)
    rater = models.resnet18(weights=None)
    num_ftrs = rater.fc.in_features
    rater.fc = nn.Linear(num_ftrs, 1)
    try:
        rater.load_state_dict(torch.load("best_face_rater_colab.pth", map_location=device))
    except FileNotFoundError:
        st.error("⚠️ Model file missing. Upload 'best_face_rater_colab.pth'.")
        return None, None
    rater.eval()

    # Segmentation Model (DeepLabV3)
    seg_model = models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
    seg_model.eval()
    
    return rater, seg_model

rater_model, seg_model = load_models()

# --- 3. PROCESSING LOGIC ---
def isolate_face_pixels(image):
    # Prepare for DeepLabV3
    seg_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = seg_transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = seg_model(input_tensor)['out'][0]
    
    output_predictions = output.argmax(0)
    # Class 15 is Person
    mask = (output_predictions == 15).byte().numpy()
    
    image_resized = image.resize((224, 224))
    img_np = np.array(image_resized)
    
    # Apply Mask (Black Background)
    mask_3d = np.stack([mask, mask, mask], axis=2)
    foreground = img_np * mask_3d
    
    return Image.fromarray(foreground)

def crop_to_face_strict(image_pil):
    img_np = np.array(image_pil)
    if len(img_np.shape) == 2: img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    
    # Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0: return image_pil, False
    
    # Largest Face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    
    # Margin logic
    margin = int(h * 0.20)
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(img_np.shape[1] - x, w + 2*margin)
    h = min(img_np.shape[0] - y, h + 2*margin)
    
    return image_pil.crop((x, y, x+w, y+h)), True

# Grad-CAM Setup
gradients = None
activations = None
def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]
def forward_hook(module, input, output):
    global activations
    activations = output

def generate_heatmap(model, input_tensor):
    target_layer = model.layer4[-1]
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)
    
    output = model(input_tensor)
    model.zero_grad()
    output.backward()
    
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(512): activations[:, i, :, :] *= pooled_gradients[i]
    
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap.detach().numpy(), 0)
    if np.max(heatmap) > 0: heatmap /= np.max(heatmap)
    
    handle_f.remove(); handle_b.remove()
    return heatmap

def overlay_heatmap(heatmap, original_image):
    heatmap = cv2.resize(heatmap, (original_image.width, original_image.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img_np = np.array(original_image)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    superimposed_img = heatmap * 0.4 + img_np
    return Image.fromarray(cv2.cvtColor(np.uint8(superimposed_img), cv2.COLOR_BGR2RGB))

# --- 4. MAIN INTERFACE ---

uploaded_file = st.file_uploader("Upload a clear portrait", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and rater_model:
    image = Image.open(uploaded_file).convert('RGB')
    
    # Processing Flow
    with st.spinner("Isolating facial geometry..."):
        cropped_img, found = crop_to_face_strict(image)
        final_input = isolate_face_pixels(cropped_img)

    # UI Columns
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Original', use_container_width=True)
    with col2:
        st.image(final_input, caption='AI Analysis View', use_container_width=True)

    st.write("")
    
    if st.button('Calculate Score'):
        progress_bar = st.progress(0)
        
        # 1. Transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(final_input).unsqueeze(0)
        input_tensor.requires_grad = True
        
        progress_bar.progress(60)
        
        # 2. Score
        with torch.no_grad():
            output = rater_model(input_tensor)
            score = output.item()
        
        score = max(1.0, min(5.0, score))
        
        # 3. Heatmap (Visual Reasoning)
        heatmap = generate_heatmap(rater_model, input_tensor)
        overlay = overlay_heatmap(heatmap, final_input)
        
        progress_bar.progress(100)
        
        # --- RESULTS DISPLAY ---
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Determine Color Code
        if score >= 4.0: score_color = "#4CAF50" # Green
        elif score >= 3.0: score_color = "#FF9800" # Orange
        else: score_color = "#F44336" # Red
        
        # Metric Card HTML
        st.markdown(f"""
            <div class="score-card">
                <p class="score-label">Aesthetic Rating</p>
                <h1 class="score-value" style="color: {score_color};">{score:.2f}</h1>
                <p style="margin-top: 10px; color: #666;">out of 5.0</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        st.image(overlay, caption='Feature Activation Map (Visual Reasoning)', use_container_width=True)
        
        if score >= 4.0:
            st.success("Exceptional features detected. High symmetry and proportion.")
            st.balloons()
        elif score >= 3.0:
            st.info("Strong features detected. Above average structure.")
        else:
            st.warning("Average structure detected. Lighting or angle may affect result.")