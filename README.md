# 😷 Real-Time PPE Compliance Monitor

An AI-powered computer vision pipeline that detects face mask compliance in real-time. This project uses a modern Convolutional Neural Network (CNN) face detector paired with a fine-tuned Deep Learning classifier to accurately track and evaluate Personal Protective Equipment (PPE) compliance, even under heavy occlusion.

## 📸 Demo


![With Mask Demo](<img width="1600" height="1014" alt="withMask" src="https://github.com/user-attachments/assets/a4e3fb1b-cf0b-4e56-ac8b-812c79766dfd" />)
> *Accurately detecting correctly worn KN95/Surgical masks.*

![No Mask / Incorrect Mask Demo](<img width="1600" height="1020" alt="withoutMask" src="https://github.com/user-attachments/assets/85af5372-b5c7-4648-b6b0-404564452506" />
)
> *Flagging non-compliance and incorrectly worn masks in real-time.*

## 🚀 Tech Stack
* **Python 3.12**
* **TensorFlow / Keras:** Fine-tuned `MobileNetV2` for mask classification.
* **OpenCV 5.0.0:** `YuNet` (CNN-based face detector) for real-time, robust face tracking.
* **NumPy:** Matrix and array manipulations.

## 🧠 How It Works
Traditional computer vision algorithms (like Haar Cascades) fail to detect faces when the nose and mouth are covered by masks. This project solves that bottleneck by using a two-step pipeline:

1. **Face Detection (YuNet):** A lightweight, native OpenCV CNN detects faces in the webcam feed regardless of steep angles, poor lighting, or heavy occlusion (hands, mugs, and masks). 
2. **Classification (MobileNetV2):** The detected face ROI is dynamically padded, preprocessed, and fed into a custom-trained MobileNetV2 model to classify the crop into one of three states:
   * `With Mask` (Green)
   * `Incorrect Mask` (Yellow)
   * `No Mask` (Red)

## 🛠️ Installation & Setup

**1. Clone the repository**

    git clone https://github.com/YOUR-USERNAME/RealTime-PPE-Compliance.git
    cd RealTime-PPE-Compliance

**2. Create a virtual environment (Recommended)**

    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate

**3. Install dependencies**

    pip install -r requirements.txt

**4. Download the YuNet ONNX Model**
Because of GitHub file size limits, you must download the OpenCV YuNet weights directly:

    curl -L -o face_detection_yunet_2026may.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx

## 💻 Usage
Run the main application script to start the webcam feed:

    python3 app.py
    
*Press **`q`** on your keyboard to quit the video stream.*

## 📁 Repository Structure
* `model_training.ipynb`: The Jupyter Notebook used to train and fine-tune the MobileNetV2 model on the FMD Dataset.
* `app.py`: The main OpenCV deployment script for real-time webcam inference.
* `ppe_mask_model.keras`: The saved weights of the trained classification model.
* `requirements.txt`: Python package dependencies.

## 👨‍💻 Author
**Rudra Pratap Singh**
