"""
NeuroScan AI - Verification & Health Check Script
Verifies model loading, dependencies, and inference on sample MRI scans.
"""

import sys
import time
from pathlib import Path

def run_health_check():
    print("=" * 60)
    print("🧠 NeuroScan AI - Model & Environment Verification")
    print("=" * 60)

    # 1. Environment and Dependencies
    print("\n[1/4] Checking Core Dependencies...")
    try:
        import os
        base_dir = Path(__file__).parent.resolve()
        ultralytics_cfg = base_dir / ".ultralytics"
        ultralytics_cfg.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(ultralytics_cfg)

        import numpy as np
        import pandas as pd
        import PIL
        import cv2
        import torch
        import streamlit
        from ultralytics import YOLO
        print(f"  ✓ Python:       {sys.version.split()[0]}")
        print(f"  ✓ PyTorch:      {torch.__version__} (Device: {'cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'})")
        print(f"  ✓ NumPy:        {np.__version__}")
        print(f"  ✓ OpenCV:       {cv2.__version__}")
        print(f"  ✓ Streamlit:    {streamlit.__version__}")
        print(f"  ✓ Pillow:       {PIL.__version__}")
    except ImportError as e:
        print(f"  ✗ Import Error: {e}")
        return False

    # 2. Check Model Checkpoint
    print("\n[2/4] Checking Model Weights...")
    base_dir = Path(__file__).parent
    model_path = base_dir / "best.pt"
    if not model_path.exists():
        print(f"  ✗ Model file not found at: {model_path}")
        return False

    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Found 'best.pt' ({size_mb:.2f} MB)")

    try:
        import os
        os.environ["YOLO_CONFIG_DIR"] = str(base_dir / ".ultralytics")
        model = YOLO(str(model_path))
        print(f"  ✓ Model Loaded Successfully!")
        print(f"  ✓ Detected Task: {model.task}")
        print(f"  ✓ Model Classes: {model.names}")
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        return False

    # 3. Check Inference on Sample Scans
    print("\n[3/4] Running Inference on Sample MRI Scans...")
    samples_dir = base_dir / "samples"
    sample_files = list(samples_dir.glob("*.jpg"))
    if not sample_files:
        print("  ⚠️ No sample images found in samples/. Testing dummy image...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        res = model.predict(dummy, verbose=False)
        print("  ✓ Dummy inference completed.")
    else:
        for sample_path in sorted(sample_files):
            start = time.time()
            res = model.predict(str(sample_path), conf=0.20, verbose=False)[0]
            elapsed_ms = (time.time() - start) * 1000
            boxes = res.boxes
            det_count = len(boxes) if boxes is not None else 0
            classes = [model.names[int(b.cls[0])] for b in boxes] if det_count else ["None"]
            print(f"  ✓ {sample_path.name:<25} | Detections: {det_count} ({', '.join(classes)}) | Latency: {elapsed_ms:.1f}ms")

    # 4. Summary
    print("\n[4/4] Verification Summary")
    print("  ✓ All health checks passed successfully!")
    print("  ✓ Application is ready to run: streamlit run app.py")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = run_health_check()
    sys.exit(0 if success else 1)
