"""
==============================================================================
NeuroScan AI — Brain Tumor Detection & Localization (YOLOv8)
==============================================================================
A robust, medical-grade diagnostic assistance interface for automated brain
tumor detection, classification, and localization using deep learning.

Supported Classes:
  - 0: Glioma
  - 1: Meningioma
  - 2: Pituitary
  - 3: No Tumor
==============================================================================
"""

import io
import os
import json
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# -----------------------------------------------------------------------------
# CONFIGURATION & ENVIRONMENT SETUP
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.resolve()

# Direct Ultralytics to store local config in workspace to prevent OS permission errors
ULTRALYTICS_DIR = BASE_DIR / ".ultralytics"
ULTRALYTICS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_DIR))

# Lazy import YOLO after setting environment
from ultralytics import YOLO

APP_TITLE = "NeuroScan AI"
APP_SUBTITLE = "Automated Brain Tumor Detection & Localization System"
MODEL_PATH = BASE_DIR / "best.pt"
SAMPLES_DIR = BASE_DIR / "samples"
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "neuroscan_logs.csv"

# Color mappings for classes (RGBA compatible RGB tuples & Hex)
CLASS_METADATA = {
    0: {"name": "Glioma", "color": (59, 130, 246), "hex": "#3B82F6", "badge": "primary"},
    1: {"name": "Meningioma", "color": (16, 185, 129), "hex": "#10B981", "badge": "success"},
    2: {"name": "Pituitary", "color": (245, 158, 11), "hex": "#F59E0B", "badge": "warning"},
    3: {"name": "No Tumor", "color": (107, 114, 128), "hex": "#6B7280", "badge": "secondary"},
}
DEFAULT_COLOR = (168, 85, 247)  # Purple for unknown classes

# Available demo samples
DEMO_SAMPLES = {
    "Glioma Tumor (Abnormal)": SAMPLES_DIR / "glioma_sample.jpg",
    "Meningioma Tumor (Abnormal)": SAMPLES_DIR / "meningioma_sample.jpg",
    "Pituitary Tumor (Abnormal)": SAMPLES_DIR / "pituitary_sample.jpg",
    "Healthy Brain (Normal Control)": SAMPLES_DIR / "healthy_sample.jpg",
}

# -----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{APP_TITLE} — Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM CSS STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Metric Card Enhancements */
    div[data-testid="stMetric"] {
        background-color: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94A3B8;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    
    /* Disclaimer Banner */
    .disclaimer-banner {
        background: linear-gradient(90deg, #1e1b4b 0%, #311042 100%);
        border-left: 5px solid #EAB308;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 20px;
        color: #FEF08A;
        font-size: 0.92rem;
        line-height: 1.5;
    }
    .disclaimer-title {
        font-weight: 700;
        color: #FACC15;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Class Legend Badges */
    .legend-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin: 12px 0 20px 0;
    }
    .legend-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        background-color: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #E2E8F0;
    }
    .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
    }

    /* Findings Cards */
    .finding-alert-positive {
        background-color: rgba(239, 68, 68, 0.12);
        border-left: 4px solid #EF4444;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 12px;
        color: #FCA5A5;
        font-weight: 500;
    }
    .finding-alert-negative {
        background-color: rgba(16, 185, 129, 0.12);
        border-left: 4px solid #10B981;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 12px;
        color: #6EE7B7;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# LOGGING & MODEL MANAGEMENT
# -----------------------------------------------------------------------------
def init_logging() -> None:
    """Ensure logs directory exists."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_run(data: dict) -> bool:
    """Append inference record to persistent CSV log."""
    try:
        init_logging()
        df = pd.DataFrame([data])
        header = not LOG_FILE.exists()
        df.to_csv(LOG_FILE, mode="a", header=header, index=False)
        return True
    except Exception as e:
        st.error(f"Failed to write log: {e}")
        return False


def load_logs() -> pd.DataFrame:
    """Read CSV log into DataFrame."""
    if LOG_FILE.exists():
        try:
            return pd.read_csv(LOG_FILE)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_model(weights_path: Path) -> Optional[YOLO]:
    """Load and cache YOLO model checkpoint."""
    if not weights_path.exists():
        return None
    try:
        return YOLO(str(weights_path))
    except Exception as e:
        st.error(f"Error initializing YOLO model: {e}")
        return None


# -----------------------------------------------------------------------------
# IMAGE ANNOTATION & RENDERING
# -----------------------------------------------------------------------------
def get_scalable_font(font_size: int) -> ImageFont.ImageFont:
    """Find and return high-contrast TrueType font across OS platforms."""
    candidate_paths = [
        # macOS
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        # Windows
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size)
            except Exception:
                continue

    # Fallback to Pillow's load_default with size if supported
    try:
        return ImageFont.load_default(size=font_size)
    except TypeError:
        return ImageFont.load_default()


def draw_detections(
    image: Image.Image,
    detections: List[Tuple[float, float, float, float, int, float]],
    names_map: Dict[int, str],
    show_boxes: bool = True,
    fill_opacity: int = 50,
) -> Image.Image:
    """
    Render high-visibility bounding boxes and labels onto the image.
    """
    if not show_boxes or not detections:
        return image

    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Dynamic font sizing based on image dimensions
    font_size = max(14, int(min(image.width, image.height) * 0.03))
    font = get_scalable_font(font_size)

    for x1, y1, x2, y2, cls_id, conf in detections:
        meta = CLASS_METADATA.get(cls_id, {"color": DEFAULT_COLOR, "name": names_map.get(cls_id, f"Class {cls_id}")})
        color_rgb = meta["color"]
        class_name = meta["name"]

        fill_color = color_rgb + (fill_opacity,)
        stroke_color = color_rgb + (255,)

        # Draw bounding rectangle
        draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=stroke_color, width=3)

        # Label styling
        label_text = f" {class_name} : {conf:.1%} "
        try:
            bbox = draw.textbbox((x1, y1), label_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w, text_h = len(label_text) * (font_size * 0.6), font_size + 4

        # Position label above or inside box
        pad = 4
        if y1 - text_h - (pad * 2) > 0:
            box_coords = [x1, y1 - text_h - (pad * 2), x1 + text_w, y1]
            text_coords = (x1, y1 - text_h - pad)
        else:
            box_coords = [x1, y1, x1 + text_w, y1 + text_h + (pad * 2)]
            text_coords = (x1, y1 + pad)

        draw.rectangle(box_coords, fill=stroke_color)
        draw.text(text_coords, label_text, fill=(255, 255, 255, 255), font=font)

    return Image.alpha_composite(base, overlay).convert("RGB")


# -----------------------------------------------------------------------------
# INFERENCE PIPELINE
# -----------------------------------------------------------------------------
def run_inference(
    model: YOLO,
    image: Image.Image,
    conf_threshold: float,
    iou_threshold: float,
) -> Tuple[List[Tuple[float, float, float, float, int, float]], Dict[str, int], float]:
    """Execute model prediction and return bounding boxes, class counts, and latency."""
    img_array = np.array(image)
    start_time = time.time()
    results = model.predict(
        img_array,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )
    inference_time = (time.time() - start_time) * 1000

    detections = []
    class_counts = defaultdict(int)

    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            coords = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = CLASS_METADATA.get(cls_id, {}).get("name", model.names.get(cls_id, f"Class {cls_id}"))

            detections.append((coords[0], coords[1], coords[2], coords[3], cls_id, conf))
            class_counts[class_name] += 1

    return detections, dict(class_counts), inference_time


# -----------------------------------------------------------------------------
# MAIN APPLICATION
# -----------------------------------------------------------------------------
def main():
    init_logging()

    # --- TOP HEADER ---
    st.markdown(f"# 🧠 {APP_TITLE}")
    st.caption(f"{APP_SUBTITLE} • Deep Learning Engine (YOLOv8)")

    # Medical Disclaimer Banner
    st.markdown("""
    <div class="disclaimer-banner">
        <div class="disclaimer-title">⚠️ RESEARCH USE ONLY — CLINICAL DISCLAIMER</div>
        NeuroScan AI is an investigative computer-aided detection (CAD) tool developed for academic research. 
        It is <strong>not</strong> an FDA/CE cleared diagnostic device. Output predictions must always be verified 
        by a licensed radiologist or medical professional before making any clinical decisions.
    </div>
    """, unsafe_allow_html=True)

    # Class Legend Pills
    legend_html = '<div class="legend-container">'
    for cid, meta in CLASS_METADATA.items():
        legend_html += f'<span class="legend-pill"><span class="legend-dot" style="background-color: {meta["hex"]};"></span>{meta["name"]}</span>'
    legend_html += '</div>'
    st.markdown(legend_html, unsafe_allow_html=True)

    # --- SIDEBAR CONFIGURATION ---
    with st.sidebar:
        st.header("⚙️ Model Controls")

        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.05,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Minimum probability score required to classify a region as a tumor."
        )

        iou_threshold = st.slider(
            "IoU Threshold (NMS)",
            min_value=0.1,
            max_value=0.9,
            value=0.45,
            step=0.05,
            help="Intersection-over-Union threshold for Non-Maximum Suppression to filter overlapping boxes."
        )

        st.subheader("🎨 Visualization")
        show_boxes = st.toggle("Overlay Bounding Boxes", value=True)
        fill_opacity = st.slider("Fill Highlight Opacity", 10, 150, 60, 10, help="Transparency of bounding box highlight.")

        st.divider()
        st.subheader("📋 Case Metadata (Optional)")
        case_id = st.text_input("Patient / Case ID", placeholder="e.g. NS-2026-081")
        scan_plane = st.selectbox("MRI Scan Plane", ["Axial", "Coronal", "Sagittal", "Unknown"])
        notes = st.text_area("Clinical Notes", placeholder="e.g., T1 post-contrast axial slice...", height=70)

        st.divider()
        st.caption(f"Weights: `{MODEL_PATH.name}`")
        if MODEL_PATH.exists():
            st.success("Checkpoint: Ready", icon="✅")
        else:
            st.error("Checkpoint: Missing", icon="❌")

    # Load Model Checkpoint
    model = load_model(MODEL_PATH)
    if model is None:
        st.error(
            f"❌ Model checkpoint not found at `{MODEL_PATH}`.\n\n"
            "Please ensure `best.pt` is present in the application root directory."
        )
        st.stop()

    # --- INPUT SELECTION TABS ---
    input_tab1, input_tab2 = st.tabs(["📁 Upload Your MRI Scan", "🧪 Test Built-in Sample Scans"])

    selected_image: Optional[Image.Image] = None
    image_source_name: str = ""

    with input_tab1:
        uploaded_file = st.file_uploader(
            "Choose a brain MRI image file",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            help="Upload an axial, coronal, or sagittal MRI slice."
        )
        if uploaded_file is not None:
            try:
                selected_image = Image.open(uploaded_file).convert("RGB")
                image_source_name = uploaded_file.name
            except Exception as e:
                st.error(f"Error opening image: {e}")

    with input_tab2:
        st.write("Quickly test the detection model using curated, verified MRI scan samples:")
        sample_choice = st.selectbox(
            "Select Sample MRI Scan",
            options=list(DEMO_SAMPLES.keys()),
            index=0
        )
        sample_path = DEMO_SAMPLES[sample_choice]
        if sample_path.exists():
            if st.button("Load This Sample", type="secondary", use_container_width=True) or (uploaded_file is None and "sample_loaded" not in st.session_state):
                st.session_state["sample_loaded"] = sample_choice

        if "sample_loaded" in st.session_state and uploaded_file is None:
            chosen = DEMO_SAMPLES[st.session_state["sample_loaded"]]
            if chosen.exists():
                selected_image = Image.open(chosen).convert("RGB")
                image_source_name = chosen.name

    # --- INFERENCE & RESULTS PRESENTATION ---
    if selected_image is not None:
        st.divider()

        # Run inference pipeline
        detections, class_counts, inference_time = run_inference(
            model,
            selected_image,
            conf_threshold,
            iou_threshold
        )

        # Annotated image generation
        annotated_image = draw_detections(
            selected_image,
            detections,
            model.names,
            show_boxes=show_boxes,
            fill_opacity=fill_opacity
        )

        col_img, col_metrics = st.columns([1.1, 0.9], gap="large")

        # --- LEFT COLUMN: VISUAL OUTPUT ---
        with col_img:
            view_mode = st.radio(
                "Image View Mode",
                ["Annotated Detection", "Original Scan", "Side-by-Side Comparison"],
                horizontal=True
            )

            if view_mode == "Annotated Detection":
                st.image(
                    annotated_image,
                    caption=f"{image_source_name} — {len(detections)} detection(s) at conf ≥ {conf_threshold:.0%}",
                    use_container_width=True
                )
            elif view_mode == "Original Scan":
                st.image(
                    selected_image,
                    caption=f"{image_source_name} — Original Scan",
                    use_container_width=True
                )
            else:
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("Original Scan")
                    st.image(selected_image, use_container_width=True)
                with c2:
                    st.caption("Annotated Findings")
                    st.image(annotated_image, use_container_width=True)

            # Download Annotated Image
            buf = io.BytesIO()
            annotated_image.save(buf, format="PNG")
            st.download_button(
                label="⬇️ Download Annotated Scan (PNG)",
                data=buf.getvalue(),
                file_name=f"neuroscan_{Path(image_source_name).stem}_annotated.png",
                mime="image/png",
                use_container_width=True
            )

        # --- RIGHT COLUMN: DIAGNOSTIC REPORT ---
        with col_metrics:
            st.subheader("📊 Diagnostic Summary")

            # Primary Diagnosis Banner
            abnormal_detections = [d for d in detections if d[4] in (0, 1, 2)]
            if abnormal_detections:
                tumor_types = list({CLASS_METADATA[d[4]]["name"] for d in abnormal_detections})
                st.markdown(f"""
                <div class="finding-alert-positive">
                    🚨 <strong>Abnormal Findings Detected:</strong> {', '.join(tumor_types)} lesion(s) localized with active confidence.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="finding-alert-negative">
                    ✅ <strong>No Abnormal Tumor Lesions Detected</strong> above threshold.
                </div>
                """, unsafe_allow_html=True)

            # Quick Metric Counters
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Detections", len(detections))
            m2.metric("Latency", f"{inference_time:.0f} ms")
            max_conf = max([d[5] for d in detections]) if detections else 0.0
            m3.metric("Max Conf", f"{max_conf:.1%}")
            m4.metric("Scan Size", f"{selected_image.width}×{selected_image.height}")

            # Findings Breakdown Table
            st.markdown("#### 🔬 Identified Regions")
            if detections:
                table_rows = []
                for idx, (x1, y1, x2, y2, cid, conf) in enumerate(detections, start=1):
                    cname = CLASS_METADATA.get(cid, {}).get("name", model.names.get(cid, f"Class {cid}"))
                    area = int((x2 - x1) * (y2 - y1))
                    table_rows.append({
                        "#": idx,
                        "Tumor Class": cname,
                        "Confidence": f"{conf:.2%}",
                        "Bounding Box [X1, Y1, X2, Y2]": f"[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]",
                        "Area (px²)": f"{area:,}",
                    })
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No suspicious regions identified above the selected threshold.")

            # Logging & Export Actions
            st.markdown("#### 💾 Audit & Export")
            col_save, col_export = st.columns(2)

            with col_save:
                if st.button("📝 Save to Audit Log", use_container_width=True):
                    record = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "case_id": case_id.strip() if case_id else "N/A",
                        "scan_plane": scan_plane,
                        "file_name": image_source_name,
                        "detections_count": len(detections),
                        "findings": json.dumps(class_counts),
                        "max_confidence": f"{max_conf:.3f}",
                        "latency_ms": f"{inference_time:.1f}",
                        "notes": notes.strip() if notes else "N/A"
                    }
                    if log_run(record):
                        st.success("Scan saved to audit log!", icon="✅")

            with col_export:
                report_data = {
                    "application": APP_TITLE,
                    "generated_at": datetime.now().isoformat(),
                    "case_id": case_id or "UNSPECIFIED",
                    "scan_plane": scan_plane,
                    "file_name": image_source_name,
                    "metrics": {
                        "detections_count": len(detections),
                        "max_confidence": float(f"{max_conf:.4f}"),
                        "inference_latency_ms": float(f"{inference_time:.2f}")
                    },
                    "detections": [
                        {
                            "id": i + 1,
                            "class": CLASS_METADATA.get(d[4], {}).get("name", model.names.get(d[4])),
                            "confidence": round(d[5], 4),
                            "box": [round(coord, 1) for coord in d[:4]]
                        }
                        for i, d in enumerate(detections)
                    ]
                }
                st.download_button(
                    label="📄 Export Report (JSON)",
                    data=json.dumps(report_data, indent=2),
                    file_name=f"neuroscan_{Path(image_source_name).stem}_report.json",
                    mime="application/json",
                    use_container_width=True
                )
    else:
        st.info("👆 Please upload an MRI scan image above or select a sample scan to view predictions.")

    # --- LOG HISTORY & AUDIT TRAIL EXPANDER ---
    st.divider()
    with st.expander("📜 Audit Logs & Historical Records"):
        logs_df = load_logs()
        if not logs_df.empty:
            st.dataframe(logs_df, use_container_width=True)
            col_csv, col_clr = st.columns([0.8, 0.2])
            with col_csv:
                csv_buffer = logs_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Audit History (CSV)",
                    data=csv_buffer,
                    file_name="neuroscan_history.csv",
                    mime="text/csv"
                )
            with col_clr:
                if st.button("🗑️ Clear Log History"):
                    if LOG_FILE.exists():
                        LOG_FILE.unlink()
                        st.rerun()
        else:
            st.caption("No audit logs recorded yet. Run inference and click 'Save to Audit Log'.")


if __name__ == "__main__":
    main()
