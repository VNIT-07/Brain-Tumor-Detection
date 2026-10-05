# 🧠 NeuroScan AI — Brain Tumor Detection & Localization (YOLOv8)

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/YOLOv8-Ultralytics-00599C?logo=yolo&logoColor=white)](https://docs.ultralytics.com/)
[![Deep Learning](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent, interactive computer-aided detection (CAD) application for automated localization and classification of brain tumors from MRI scans using a fine-tuned **YOLOv8** deep learning architecture.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Supported Tumor Classes](#supported-tumor-classes)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
  - [Option A: One-Click Launch (Recommended)](#option-a-one-click-launch-recommended)
  - [Option B: Manual Setup](#option-b-manual-setup)
- [Health Check & Verification](#health-check--verification)
- [Dataset Information](#dataset-information)
- [Audit Logging & Export](#audit-logging--export)
- [Troubleshooting & Dependency Notes](#troubleshooting--dependency-notes)
- [Clinical Disclaimer](#clinical-disclaimer)
- [License](#license)

---

## 🌟 Overview

**NeuroScan AI** provides researchers and medical imaging students with a seamless, turnkey platform to upload brain MRI scans, perform real-time neural inference, visualize bounding-box localization with confidence scoring, and export structured clinical audit records.

The pipeline is pre-configured with trained weights (`best.pt`) and includes verified sample MRI slices for immediate one-click testing out of the box.

---

## ✨ Key Features

- **High-Accuracy YOLOv8 Inference**: Real-time object detection model checkpoint (`best.pt`) optimized for brain MRI slices.
- **🧪 1-Click Built-in Demo Scans**: Test the application immediately with verified sample scans for each tumor class (Glioma, Meningioma, Pituitary, and Normal Healthy brain).
- **🎨 Interactive Visualization**:
  - Color-coded bounding boxes with class confidence labels.
  - Dynamic opacity control for lesion highlight masks.
  - Three viewing modes: *Annotated Detection*, *Original Scan*, and *Side-by-Side Comparison*.
- **📊 Quantitative Findings**:
  - Detections counter, inference latency in milliseconds, and peak confidence score.
  - Detailed findings table listing bounding box coordinates `[X1, Y1, X2, Y2]` and lesion area in pixels.
- **💾 Comprehensive Export Options**:
  - Download high-resolution annotated MRI scans (PNG).
  - Export machine-readable clinical reports (JSON).
  - Append runs to an audit trail and download history logs (CSV).
- **🛡️ Cross-Platform Portability**: Tested and configured for macOS, Linux, and Windows with zero missing dependency crashes.

---

## 🏷️ Supported Tumor Classes

| Class ID | Tumor Type | Color Coding | Clinical Description |
| :---: | :---: | :---: | :--- |
| **0** | **Glioma** | 🟦 Blue (`#3B82F6`) | Tumors originating in the glial cells of the brain or spine |
| **1** | **Meningioma** | 🟩 Green (`#10B981`) | Typically benign tumors arising from the meninges |
| **2** | **Pituitary** | 🟧 Amber (`#F59E0B`) | Abnormal growths developing in the pituitary gland |
| **3** | **No Tumor** | ⬜ Slate (`#6B7280`) | Normal brain tissue control / healthy scans |

---

## 📁 Project Structure

```text
Brain-Tumor-Detection/
├── .streamlit/
│   ├── config.toml           # Streamlit theme and server configuration
│   └── credentials.toml      # Disables initial onboarding prompt
├── logs/
│   └── neuroscan_logs.csv    # Persistent inference audit logs
├── samples/                  # Curated MRI sample scans for instant testing
│   ├── glioma_sample.jpg
│   ├── healthy_sample.jpg
│   ├── meningioma_sample.jpg
│   └── pituitary_sample.jpg
├── .gitignore                # Clean git exclusion rules
├── LICENSE                   # MIT License
├── README.md                 # Project documentation
├── app.py                    # Streamlit web application
├── best.pt                   # Trained YOLOv8 checkpoint weights
├── requirements.txt          # Python dependency specifications
├── run.bat                   # 1-Click Windows launcher script
├── run.sh                    # 1-Click macOS/Linux launcher script
└── test_model.py             # CLI validation & health check script
```

---

## 🚀 Quick Start

### Option A: One-Click Launch (Recommended)

#### On macOS / Linux:
```bash
./run.sh
```
*(The script will automatically set up the virtual environment, install requirements if needed, and start the app).*

#### On Windows:
Double-click `run.bat` or run:
```cmd
run.bat
```

The application will launch and open in your default browser at `http://localhost:8501`.

---

### Option B: Manual Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/VNIT-07/Brain-Tumor-Detection.git
   cd Brain-Tumor-Detection
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate

   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Launch the web application:**
   ```bash
   streamlit run app.py
   ```

---

## 🩺 Health Check & Verification

You can verify that your Python environment, PyTorch installation, model weights, and inference pipelines are working properly at any time by running the built-in CLI test:

```bash
python test_model.py
```

Expected output:
```text
============================================================
🧠 NeuroScan AI - Model & Environment Verification
============================================================
[1/4] Checking Core Dependencies...
  ✓ Python:       3.11.x
  ✓ PyTorch:      2.2.2
  ✓ NumPy:        1.26.4
  ✓ OpenCV:       4.10.0
  ✓ Streamlit:    1.x
  ✓ Pillow:       12.x
[2/4] Checking Model Weights...
  ✓ Found 'best.pt' (21.48 MB)
  ✓ Model Loaded Successfully!
  ✓ Model Classes: {0: 'Glioma', 1: 'Meningioma', 2: 'Pituitary', 3: 'No Tumor'}
[3/4] Running Inference on Sample MRI Scans...
  ✓ glioma_sample.jpg         | Detections: 1 (Glioma)
  ✓ healthy_sample.jpg        | Detections: 0 (None)
  ✓ meningioma_sample.jpg     | Detections: 1 (Meningioma)
  ✓ pituitary_sample.jpg      | Detections: 1 (Pituitary)
[4/4] Verification Summary
  ✓ All health checks passed successfully!
============================================================
```

---

## 📊 Dataset Information

- **Dataset Name:** Medical Image Dataset: Brain Tumor Detection
- **Source:** [Kaggle Dataset Repository](https://www.kaggle.com/datasets/pkdarabi/medical-image-dataset-brain-tumor-detection)
- **Modalities:** High-resolution T1/T2 MRI brain slices formatted for object detection and multi-class classification.
- **Classes:** Glioma, Meningioma, Pituitary Adenoma, and Healthy Brain Tissue.

---

## 📋 Audit Logging & Export

Every scan processed in the user interface can be logged with one click using the **"Save to Audit Log"** button.

### Logged Fields
- `timestamp`: Date and time of evaluation
- `case_id`: Optional anonymized patient/case identifier
- `scan_plane`: MRI slice orientation (Axial, Coronal, Sagittal)
- `file_name`: Original uploaded image file name
- `detections_count`: Number of localized lesions
- `findings`: JSON-encoded distribution of detected tumor types
- `max_confidence`: Highest detection probability score
- `latency_ms`: Neural inference duration in milliseconds
- `notes`: Custom radiologist / researcher observations

Stored in `logs/neuroscan_logs.csv` and directly viewable/downloadable inside the web application under the **Audit Logs & Historical Records** panel.

---

## 🔧 Troubleshooting & Dependency Notes

> [!IMPORTANT]
> **NumPy Compatibility Note (`numpy<2.0.0`):**
> PyTorch versions prior to 2.4 (including PyTorch 2.2.2 on macOS x86_64) require NumPy 1.x due to C-API ABI changes introduced in NumPy 2.x. If you encounter `RuntimeError: Numpy is not available` or `_ARRAY_API not found`, ensure that your environment uses `numpy<2.0.0` as pinned in `requirements.txt`.

If you experience permission issues with Ultralytics configuration directories on managed machines, `app.py` and `run.sh` automatically route settings to a local `.ultralytics` workspace folder.

---

## ⚖️ Clinical Disclaimer

> [!CAUTION]
> **FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY.**
> NeuroScan AI is an investigative deep learning demonstration tool. It has **not** been cleared or approved by any medical device regulatory agency (such as the US FDA or EMA). Do not use this tool as a primary diagnostic device or as a substitute for professional clinical radiological evaluation.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) — see the LICENSE file for details.
