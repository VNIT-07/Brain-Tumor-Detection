
# import streamlit as st
# from ultralytics import YOLO
# from PIL import Image
# import numpy as np

# # --- PAGE CONFIG ---
# st.set_page_config(page_title="NeuroScan AI (Detection)", page_icon="🧠", layout="wide")

# # --- LOAD MODEL ---
# # This expects the 'best.pt' file to be in the SAME folder as this script
# @st.cache_resource
# def load_model():
#     try:
#         model = YOLO('best.pt')
#         return model
#     except Exception as e:
#         return None

# # Load the model immediately
# model = load_model()

# # --- UI HEADER ---
# st.title("🧠 NeuroScan: Brain Tumor Detection")
# st.markdown("### Powered by YOLOv8")
# st.info("Upload an MRI scan. The AI will detect the tumor location and classify the type.")

# # --- SIDEBAR ---
# with st.sidebar:
#     st.header("Control Panel")
#     confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
#     st.write("Adjust this if the AI is missing tumors (lower it) or seeing fake ones (raise it).")
    
#     st.divider()
#     st.write("*Detected Classes:*")
#     st.markdown("- Glioma\n- Meningioma\n- Pituitary\n- No Tumor")

# # --- MAIN APP ---
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("Upload Scan")
#     uploaded_file = st.file_uploader("Choose an MRI image...", type=['jpg', 'jpeg', 'png'])
    
#     if uploaded_file:
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Original Image", use_column_width=True)

# with col2:
#     st.subheader("AI Analysis")
    
#     if uploaded_file and model:
#         if st.button("Analyze Scan", type="primary"):
#             with st.spinner("Scanning for anomalies..."):
#                 # 1. Run Inference
#                 # conf=threshold sets how strict the AI is
#                 results = model.predict(image, conf=confidence_threshold)
                
#                 # 2. Visualize
#                 # plot() returns a numpy array with the boxes drawn
#                 res_plotted = results[0].plot()
                
#                 # 3. Display
#                 st.image(res_plotted, caption="Detected Anomalies", use_column_width=True)
                
#                 # 4. Text Summary
#                 st.divider()
#                 st.write("*Detailed Findings:*")
                
#                 boxes = results[0].boxes
#                 if len(boxes) == 0:
#                     st.success("No tumors detected (Healthy).")
#                 else:
#                     for box in boxes:
#                         # Get Class Name
#                         class_id = int(box.cls[0])
#                         class_name = model.names[class_id]
                        
#                         # Get Confidence
#                         conf = float(box.conf[0])
                        
#                         # Display
#                         if class_name == "No Tumor":
#                              st.success(f"• {class_name} ({conf*100:.1f}%)")
#                         else:
#                              st.error(f"• Detected: *{class_name}* ({conf*100:.1f}%)")

#     elif not model:
#         st.error("⚠ Model file 'best.pt' not found!")
#         st.write("Please place the 'best.pt' file (downloaded from training) in this folder.")


import io
import json
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# -------------------------------------------------------
# CONFIGURATION & CONSTANTS
# -------------------------------------------------------
APP_TITLE = "NeuroScan AI"

# Use a relative path for portability across machines/cloud
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "neuroscan_logs.csv"
MODEL_PATH = BASE_DIR / "best.pt"

# Standard colors for specific classes (consistent UI)
CLASS_COLORS = {
    0: (59, 130, 246),   # Glioma: Blue
    1: (16, 185, 129),   # Meningioma: Green
    2: (245, 158, 11),   # Pituitary: Orange
    3: (107, 114, 128),  # No Tumor: Gray
}
DEFAULT_COLOR = (255, 0, 255)  # Magenta for unknown

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------
# CSS STYLING
# -------------------------------------------------------
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0; }
    .sub-header { font-size: 1.1rem; color: #6B7280; margin-bottom: 2rem; }
    .warning-box { 
        background-color: #2e1056; 
        color: #FDE047; 
        border-left: 6px solid #FACC15; 
        padding: 1.2rem; 
        border-radius: 8px; 
        margin-bottom: 1.5rem; 
        font-weight: 500;
    }
    .stat-box { background-color: #F3F4F6; padding: 1rem; border-radius: 8px; text-align: center; box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05); }
    .stat-value { font-size: 1.5rem; font-weight: 700; color: #1F2937; }
    .stat-label { font-size: 0.875rem; color: #6B7280; text-transform: uppercase; letter-spacing: 0.05em; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# UTILITY FUNCTIONS
# -------------------------------------------------------
def init_logging():
    """Ensure log directory exists."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

def log_run(data: dict):
    """Append a run dictionary to the CSV log."""
    try:
        df = pd.DataFrame([data])
        header = not LOG_FILE.exists()
        df.to_csv(LOG_FILE, mode="a", header=header, index=False)
        return True
    except Exception as e:
        st.error(f"Logging failed: {e}")
        return False

def get_color(class_id):
    """Return specific color for class or default."""
    return CLASS_COLORS.get(class_id, DEFAULT_COLOR)

@st.cache_resource
def load_yolo_model(path: Path):
    """Load YOLO model with error handling."""
    if not path.exists():
        return None
    try:
        return YOLO(str(path))
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# -------------------------------------------------------
# IMAGE PROCESSING
# -------------------------------------------------------
def draw_detections(pil_img, boxes, names_map, draw_boxes=True):
    """
    Draw overlays for detections.
    Returns: PIL Image.
    """
    if not draw_boxes or not boxes:
        return pil_img

    base = pil_img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    for (x1, y1, x2, y2, cls_id, conf) in boxes:
        color_rgb = get_color(cls_id)
        fill_color = color_rgb + (60,)    # transparent fill
        stroke_color = color_rgb + (255,) # solid border

        # Bounding box
        draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=stroke_color, width=3)

        # Label
        class_name = names_map.get(cls_id, f"Class {cls_id}")
        label_text = f"{class_name} | {conf:.0%}"

        try:
            bbox = draw.textbbox((x1, y1), label_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except:
            text_w, text_h = len(label_text) * 8, 14

        if y1 - text_h - 6 > 0:
            text_origin = (x1, y1 - text_h - 6)
        else:
            text_origin = (x1, y1 + 6)

        rect_origin = (
            text_origin[0] - 4,
            text_origin[1] - 2,
            text_origin[0] + text_w + 4,
            text_origin[1] + text_h + 4,
        )

        draw.rectangle(rect_origin, fill=stroke_color)
        draw.text(text_origin, label_text, fill=(255, 255, 255, 255), font=font)

    return Image.alpha_composite(base, overlay).convert("RGB")

# -------------------------------------------------------
# MAIN APP LAYOUT
# -------------------------------------------------------
def main():
    init_logging()

    # Header
    st.markdown(f"<div class='main-header'>{APP_TITLE}</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Automated Brain Tumor Detection & Localization</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='warning-box'>
        DISCLAIMER: RESEARCH USE ONLY.<br>
        This tool uses Artificial Intelligence (YOLOv8). Do not use results for clinical diagnosis 
        without review by a certified radiologist.
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("Configuration")

        st.subheader("Model Settings")
        conf_threshold = st.slider(
            "Confidence Threshold",
            0.0, 1.0, 0.25, 0.05,
            help="Minimum probability to count as a detection."
        )
        iou_threshold = st.slider(
            "IOU Threshold",
            0.0, 1.0, 0.45, 0.05,
            help="Intersection Over Union for filtering overlaps."
        )

        st.subheader("Visuals")
        show_boxes = st.toggle("Show Bounding Boxes", True)

        st.divider()
        st.subheader("Metadata (Optional)")
        case_id = st.text_input("Patient/Case ID", placeholder="e.g. P-1024")
        scan_plane = st.selectbox("MRI Plane", ["Axial", "Sagittal", "Coronal", "Unknown"])

    # Model Loading
    model = load_yolo_model(MODEL_PATH)
    if model is None:
        st.error(f"Model file not found at {MODEL_PATH}. Place 'best.pt' in the app directory.")
        st.stop()

    # File Upload
    uploaded_file = st.file_uploader(
        "Upload MRI Scan", 
        type=["jpg", "jpeg", "png", "bmp", "tiff"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        # Inference
        start_time = time.time()
        results = model.predict(
            img_array,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        inference_time = (time.time() - start_time) * 1000

        result = results[0]

        detections = []
        class_counts = defaultdict(int)

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                coords = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls_id]

                detections.append(
                    (coords[0], coords[1], coords[2], coords[3], cls_id, conf)
                )
                class_counts[name] += 1

        # --- DISPLAY RESULTS ---
        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader("Visual Analysis")

            if show_boxes and detections:
                annotated_img = draw_detections(image, detections, model.names)

                buf = io.BytesIO()
                annotated_img.save(buf, format="PNG")
                st.download_button(
                    label="⬇ Download Annotated Image",
                    data=buf.getvalue(),
                    file_name="neuroscan_result.png",
                    mime="image/png",
                    use_container_width=True
                )

                st.image(
                    annotated_img,
                    use_column_width=True,
                    caption=f"Processed Image ({len(detections)} detections)"
                )
            else:
                st.image(
                    image,
                    use_column_width=True,
                    caption="Original Image (No detections above threshold)"
                )

        with col2:
            st.subheader("Diagnostic Report")

            # Key Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Detections", len(detections))
            m2.metric("Latency", f"{inference_time:.0f} ms")
            highest_conf = max([d[5] for d in detections]) if detections else 0
            m3.metric("Max Conf", f"{highest_conf:.1%}")

            st.divider()

            # Detailed Findings
            if detections:
                st.write("### Findings Breakdown")
                for cls_name, count in class_counts.items():
                    st.info(f"**{cls_name}:** {count} region(s) identified")

                det_df = pd.DataFrame([
                    {
                        "Class": model.names[d[4]],
                        "Confidence": f"{d[5]:.1%}"
                    }
                    for d in detections
                ])
                st.dataframe(det_df, use_container_width=True)
            else:
                st.success("No tumor patterns detected above the selected threshold.")

            # Logging
            st.divider()
            if st.button("Save Record to Log"):
                log_data = {
                    "timestamp": datetime.now().isoformat(),
                    "case_id": case_id,
                    "scan_plane": scan_plane,
                    "file_name": uploaded_file.name,
                    "detections_count": len(detections),
                    "findings": json.dumps(class_counts),
                }
                if log_run(log_data):
                    st.toast("Run saved successfully!", icon="✅")

if __name__ == "__main__":
    main()
