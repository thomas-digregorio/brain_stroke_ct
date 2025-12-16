import streamlit as st
import requests
from PIL import Image
import io
import torch
import os
# Import local handler for "Offline/Local" mode
from handler import StrokeHandler

# Page Config
st.set_page_config(
    page_title="Brain Stroke AI Classification",
    page_icon="🧠",
    layout="centered"
)

# Custom CSS for "Premium" feel
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #ff4b4b;
        border-radius: 10px;
    }
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    h1 {
        text-align: center; 
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        font-size: 24px;
        font-weight: bold;
    }
    .safe {
        background-color: #21c354;
        color: white;
    }
    .danger {
        background-color: #ff2b2b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("🧠 NeuroScan AI")
st.markdown("### Clinical-Grade Brain Stroke Detection")
st.write("Upload a non-contrast CT slice to detect Ischemic or Hemorrhagic Stroke.")

# Sidebar
st.sidebar.header("Settings")
# mode = st.sidebar.radio("Inference Mode", ["Local (GPU)", "Cloud (Vertex AI)"])

# Sensitivity Slider
st.sidebar.markdown("---")
confidence_threshold = st.sidebar.slider("Sensitivity Threshold", 0.0, 1.0, 0.5, help="Adjust the threshold for classifying a stroke. Lower values increase sensitivity.")


st.sidebar.markdown("---")
st.sidebar.header("Sample Gallery")
st.sidebar.write("Click an image to test:")

# Hardcoded Examples from dataset
example_images = {
    "Normal 1": "assets/samples/10000.png",
    "Normal 2": "assets/samples/10005.png",
    "Normal 3": "assets/samples/10011.png",
    "Stroke 1": "assets/samples/10002.png",
    "Stroke 2": "assets/samples/10036.png",
    "Stroke 3": "assets/samples/10049.png", 
}

selected_example = None
cols = st.sidebar.columns(2)
for i, (name, path) in enumerate(example_images.items()):
    with cols[i % 2]:
        if os.path.exists(path):
            st.image(path, caption=name, width='stretch')
            if st.button(f"Load {name}", key=name):
                selected_example = path
        else:
            st.error(f"Missing: {name}")

# Initialize Handler (Singleton-ish)
@st.cache_resource
def get_handler():
    # Points to model in parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '..', 'best_model.pth')
    return StrokeHandler(model_path=model_path)

local_handler = get_handler()

# File Uploader
uploaded_file = st.file_uploader("Choose a CT Scan...", type=["jpg", "png", "jpeg"])

# Logic to load image: either uploaded or selected from gallery
image_to_process = None
caption = ""

if uploaded_file is not None:
    image_to_process = Image.open(uploaded_file)
    caption = "Uploaded CT Scan"
elif selected_example is not None:
    image_to_process = Image.open(selected_example)
    caption = f"Example: {selected_example}"

if image_to_process is not None:
    # Display Image
    st.image(image_to_process, caption=caption, width='stretch')
    
    st.write("Analyzing...")
    
    st.write("Analyzing...")
    
    # Convert to bytes for handler
    img_byte_arr = io.BytesIO()
    image_to_process.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()
    
    # Inference
    result = local_handler.handle(img_bytes, threshold=confidence_threshold)

    # Display Result
    if "error" in result:
        st.error(f"Error: {result['error']}")
    else:
        pred = result['prediction']
        conf = result['confidence']
        
        # Display logic: If Normal, show "Confidence it is Normal" (1 - p)
        # If Stroke, show "Confidence it is Stroke" (p)
        display_conf = conf
        if pred == "Normal":
            display_conf = 1.0 - conf

        # Color coding
        color_class = "danger" if pred == "Stroke" else "safe"
        
        st.markdown(f"""
        <div class="prediction-box {color_class}">
            Result: {pred.upper()} <br>
            Confidence: {display_conf:.2%}
        </div>
        """, unsafe_allow_html=True)
        
        # Clinical Context
        if pred == "Stroke":
            st.warning("⚠️ CRITICAL: Signs of stroke detected. Immediate radiologist review recommended.")
        else:
            st.success("✅ No acute stroke signs detected. Standard review protocol.")
            
        with st.expander("Show Technical Details"):
            st.json(result)
