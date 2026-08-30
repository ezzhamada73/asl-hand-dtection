# 🤟 ASL Sign Language Detection Pipeline (Streamlit Deployment)

Real-time American Sign Language (ASL) detection and classification web application powered by **YOLOv8** for hand detection and **EfficientNetB4** for sign classification.

---

## 📌 Project Overview

This repository provides a web-based deployment wrapper for an end-to-end ASL detection pipeline:
1. **Hand Detection**: YOLOv8 model (`best.pt`) detects hand bounding boxes in input frames.
2. **Crop & Padding**: Extracts detected hand region with a 20-pixel padding (`CROP_PADDING = 20`).
3. **Preprocessing**: Converts frame to RGB, resizes to 224x224, and applies `tensorflow.keras.applications.efficientnet.preprocess_input`.
4. **ASL Classification**: EfficientNetB4 model (`best_asl_model.keras`) predicts sign class out of 29 labels (`A-Z`, `del`, `nothing`, `space`).
5. **Interactive UI**: Supports real-time webcam streaming via `streamlit-webrtc` and video file uploads (`.mp4`, `.mov`, `.avi`).

---

## ⚡ Git LFS Requirements (CRITICAL)

The ASL classifier model (`best_asl_model.keras`) is **354 MB**, exceeding GitHub's 100 MB hard file limit for regular git commits. It is tracked using **Git LFS (Large File Storage)**.

### 📥 Cloning & Downloading Weight Files
If you clone this repository, you **MUST** run Git LFS commands to pull the actual model weight files instead of git pointer text files:

```bash
# 1. Install Git LFS hooks on your machine (one-time setup)
git lfs install

# 2. Clone the repository
git clone <your-repo-url>
cd <repo-folder>

# 3. Download the actual LFS tracked weight files (.keras and .pt)
git lfs pull
```

> [!CAUTION]
> If you omit `git lfs pull`, `best_asl_model.keras` will be a small (~1 KB) text pointer file, and Streamlit will fail to launch with a model deserialization error.

### ⚠️ GitHub LFS Quota & Bandwidth Warning
- **GitHub Free Tier Limits**: 1 GB total storage quota and 1 GB monthly bandwidth allowance.
- **Quota Impact**: Downloading `best_asl_model.keras` (354 MB) consumes ~35% of your total monthly free bandwidth per clone or build.
- **Best Practice**: Avoid re-pushing unnecessary model weight commits or triggering frequent clean deploy builds on Streamlit Cloud to prevent hitting monthly bandwidth caps.

---

## 🚀 Local Setup & Execution

### 1. Install Dependencies
Create a virtual environment and install required Python packages:

```bash
# Create and activate virtual environment (optional)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Desktop Pipeline (OpenCV `cv2.imshow`)
To test the desktop window pipeline locally:

```bash
python asl_yolo_pipeline_final.py
```

### 3. Run Streamlit Web Application
To start the Streamlit web application locally:

```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## ☁️ Deployment on Streamlit Community Cloud

1. Push this repository (including `.gitattributes`, `best_asl_model.keras`, `best.pt`, `app.py`, `requirements.txt`, and `packages.txt`) to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New App**, select your GitHub repository and branch.
4. Set **Main file path** to `app.py`.
5. Click **Deploy**.

> [!NOTE]
> Streamlit Cloud automatically handles Git LFS tracking during deployment cloning. Initial build and startup may take 1-2 minutes while downloading and caching the 354 MB EfficientNetB4 weight file.

---

## ⚠️ Known Limitations

- **Webcam Latency**: Live streaming performance via WebRTC depends on network stability and browser hardware acceleration.
- **Framing & Lighting Sensitivity**: The EfficientNetB4 classifier was trained on the ASL Alphabet dataset. Accuracy depends on clear hand framing and lighting similar to dataset conditions.
- **Multiple Hands**: The pipeline detects hands in order of YOLO confidence and highlights detected regions in real time.

---

## 📁 Repository Structure

```
.
├── .gitattributes          # Git LFS tracking configuration for *.keras and *.pt
├── app.py                  # Streamlit web application wrapper (Webcam + Video mode)
├── asl_yolo_pipeline_final.py # Desktop reference pipeline (OpenCV cv2.imshow)
├── best.pt                 # YOLOv8 hand detection weights (6.2 MB)
├── best_asl_model.keras    # EfficientNetB4 classifier weights (354 MB - LFS)
├── packages.txt            # System dependencies for Streamlit Cloud (libgl1, ffmpeg)
├── requirements.txt        # Python library dependencies
└── README.md               # Project documentation and deployment guide
```
