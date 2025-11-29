# **NeuroScan AI — Brain Tumor Detection (YOLOv8)**

*Streamlit-based MRI Tumor Detection Interface*


---

## **Overview**

NeuroScan AI is a research-oriented application built for automated brain tumor detection and localization using a YOLOv8 deep-learning model. The system includes MRI upload, model inference, confidence controls, visual overlays, structured reporting, and exportable logs. It is optimized for researchers studying medical imaging, tumor segmentation, and detection pipelines.

---

## **Dataset Description**

**Dataset Source:**
Kaggle — *Medical Image Dataset: Brain Tumor Detection*
[https://www.kaggle.com/datasets/pkdarabi/medical-image-dataset-brain-tumor-detection](https://www.kaggle.com/datasets/pkdarabi/medical-image-dataset-brain-tumor-detection)

**About the Dataset**
This dataset contains MRI brain images labeled into multiple tumor categories. It is intended for machine learning tasks such as classification, detection, and segmentation. The dataset includes high-resolution MRI slices categorized into:

* **Glioma Tumors**
* **Meningioma Tumors**
* **Pituitary Tumors**
* **Healthy / No Tumor**

**Why This Dataset Was Used**

* Contains clean MRI scans suitable for YOLO-based object detection.
* Provides balanced representation of major brain tumor types.
* Offers sufficient sample size for training deep-learning models.
* Public, reproducible, and structured for supervised learning.

---

## **Features**

* MRI image upload and preprocessing
* YOLOv8 inference through `best.pt`
* Color-coded bounding boxes for each tumor class
* Confidence, latency, and per-class detection metrics
* Tabular prediction breakdown
* Adjustable confidence & IoU thresholds
* Metadata fields (Case ID, MRI plane)
* Downloadable annotated images
* CSV-based logging for dataset/research tracking

---

## **Directory Structure**

```
/project-root
│
├── app.py                     # Streamlit application
├── best.pt                    # YOLOv8 trained model checkpoint
├── logs/
│   └── neuroscan_logs.csv     # Auto-generated logs
└── README.md
```

---

## Installation
---

### 1. Clone Repository

```bash
git clone https://github.com/VNIT-07/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection
```

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

---

## **Run the Application**

```bash
streamlit run app.py
```

---

## **Model Information**

### **YOLOv8 Classes**

| ID | Class Name |
| -- | ---------- |
| 0  | Glioma     |
| 1  | Meningioma |
| 2  | Pituitary  |
| 3  | No Tumor   |

### **Model File**

`best.pt` is loaded from the project root and contains the trained YOLOv8 weights.

---

## **Usage Workflow**

1. Start the Streamlit server.
2. Upload an MRI image.
3. Adjust inference thresholds if needed.
4. View bounding-box detections and confidence scores.
5. Export annotated images.
6. Log inference results for dataset tracking.

---

## **Logging System**

Inference results are appended to:

```
logs/neuroscan_logs.csv
```

### **Logged Metadata**

* Timestamp
* Patient / Case ID
* MRI Scan Plane
* Uploaded Filename
* Detection Count
* Class Distribution (JSON)

---

## **Disclaimer**

This application is **for research use only**.
Not intended for clinical or diagnostic decision-making.

---

## **License**

Specify your preferred license (MIT, Apache-2.0, etc.).
