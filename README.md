# 😊 Real-Time Emotion Detection Web Application 🎭

A deep learning-based emotion detection system that recognizes facial emotions in real-time using a webcam. Built with TensorFlow, OpenCV, and Streamlit for an interactive web interface.

![Python Version](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.54-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset Information](#dataset-information)
- [Model Architecture](#model-architecture)
- [Training Results](#training-results)
- [Installation](#installation)
- [Usage](#usage)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [License](#license)

---

## 🎯 Overview

This project implements a real-time emotion detection system capable of recognizing five distinct facial emotions: **Angry**, **Happy**, **Neutral**, **Sad**, and **Surprised**. The system uses a Convolutional Neural Network (CNN) trained on grayscale facial images and provides instant emotion predictions through a user-friendly web interface.

**Note**: This is an educational project developed as part of a machine learning course.

### Why This Project?

Emotion detection has numerous practical applications:
- **Mental Health Monitoring**: Track emotional patterns over time
- **Customer Service**: Analyze customer satisfaction in real-time
- **Education**: Monitor student engagement in virtual classrooms
- **Human-Computer Interaction**: Create emotionally-aware applications
- **Market Research**: Gauge emotional responses to products/content

---

## ✨ Features

- ✅ **Real-time emotion detection** from webcam feed
- ✅ **5 emotion categories**: Angry, Happy, Neutral, Sad, Surprised
- ✅ **Beautiful, modern UI** with dark gradient background
- ✅ **Color-coded bounding boxes** for each emotion
- ✅ **Confidence scores** displayed as percentages
- ✅ **Customizable settings** via sidebar controls
- ✅ **Cross-platform support** (Windows, Mac, Linux)

---

## 📊 Dataset Information

### Dataset Source
Custom dataset provided by course instructor for educational purposes (not included in repository).

### Dataset Statistics

| Emotion | Number of Images | Percentage |
|---------|-----------------|------------|
| **Angry** | 3,009 | 18.7% |
| **Happy** | 4,097 | 25.5% |
| **Neutral** | 3,026 | 18.8% |
| **Sad** | 3,026 | 18.8% |
| **Surprised** | 2,924 | 18.2% |
| **TOTAL** | **16,082** | **100%** |

### Dataset Characteristics
- **Image Format**: Grayscale (1 channel)
- **Image Size**: 48×48 pixels
- **Color Space**: Grayscale
- **Train/Test Split**: 80/20 ratio
  - Training samples: 12,865 images
  - Testing samples: 3,217 images
- **Preprocessing**: Images normalized to [0, 1] range

### Data Distribution

The dataset shows a relatively balanced distribution across emotions, with **Happy** being slightly overrepresented (25.5%) and **Surprised** being slightly underrepresented (18.2%).

**Note**: The dataset is NOT included in this repository due to:
- Large file size (prohibitive for Git/GitHub)
- Copyright/licensing considerations
- Course-provided data (not publicly distributable)

---

## 🏗️ Model Architecture

### Network Design

A custom **Convolutional Neural Network (CNN)** built with Keras Sequential API:

```
Model: Sequential CNN
_________________________________________________________________
Layer (type)                Output Shape              Params
=================================================================
Conv2D (3x3, 45 filters)    (None, 46, 46, 45)        450
MaxPooling2D (2x2)          (None, 23, 23, 45)        0
_________________________________________________________________
Conv2D (3x3, 65 filters)    (None, 21, 21, 65)        26,390
MaxPooling2D (2x2)          (None, 10, 10, 65)        0
_________________________________________________________________
Conv2D (3x3, 128 filters)   (None, 8, 8, 128)         74,880
MaxPooling2D (2x2)          (None, 4, 4, 128)         0
_________________________________________________________________
Flatten                     (None, 2048)              0
_________________________________________________________________
Dense (256 units, ReLU)     (None, 256)               524,544
_________________________________________________________________
Dense (5 units, Softmax)    (None, 5)                 1,285
=================================================================
Total params: 627,549
Trainable params: 627,549
Non-trainable params: 0
```

### Architecture Details

**Convolutional Layers:**
- **Layer 1**: 45 filters (3×3), ReLU activation, MaxPooling (2×2)
- **Layer 2**: 65 filters (3×3), ReLU activation, MaxPooling (2×2)
- **Layer 3**: 128 filters (3×3), ReLU activation, MaxPooling (2×2)

**Dense Layers:**
- **Hidden Layer**: 256 neurons, ReLU activation
- **Output Layer**: 5 neurons, Softmax activation (probability distribution)

### Training Configuration

- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Batch Size**: 32 (default)
- **Epochs**: 30
- **Training Environment**: Google Colab (CPU)
- **Training Duration**: ~30 minutes

---

## 📈 Training Results

### Performance Metrics

| Metric | Training Set | Validation Set |
|--------|-------------|----------------|
| **Accuracy** | 98.57% | 61.08% |
| **Loss** | 0.0478 | 3.2893 |

### Training Progress (30 Epochs)

| Epoch Range | Train Accuracy | Val Accuracy | Train Loss | Val Loss |
|-------------|---------------|--------------|------------|----------|
| 1-5 | 31% → 62% | 47% → 58% | 1.524 → 0.954 | 1.297 → 1.092 |
| 6-10 | 63% → 80% | 62% → 62% | 0.944 → 0.537 | 1.001 → 1.112 |
| 11-15 | 82% → 88% | 62% → 61% | 0.474 → 0.363 | 1.235 → 1.724 |
| 16-20 | 95% → 98% | 60% → 61% | 0.146 → 0.076 | 1.855 → 2.398 |
| 21-25 | 97% → 98% | 61% → 60% | 0.082 → 0.055 | 2.592 → 3.051 |
| 26-30 | 98% → 99% | 59% → 61% | 0.063 → 0.048 | 3.083 → 3.289 |

### Key Observations

1. **Overfitting Present**: Large gap between training (98.57%) and validation (61.08%) accuracy
2. **Early Plateau**: Validation accuracy plateaus around epoch 6-7 at ~61-62%
3. **Training Loss Continues Decreasing**: Model keeps memorizing training data
4. **Validation Loss Increases**: Clear sign of overfitting after epoch 10

### Real-World Performance by Emotion

| Emotion | Performance | Notes |
|---------|------------|-------|
| **Happy** | ⭐⭐⭐⭐⭐ Excellent | Most accurately predicted |
| **Neutral** | ⭐⭐⭐⭐⭐ Excellent | High accuracy, consistent |
| **Surprised** | ⭐⭐⭐⭐⭐ Excellent | Reliable predictions |
| **Angry** | ⭐⭐⭐ Moderate | Struggles with detection |
| **Sad** | ⭐⭐⭐ Moderate | Often confused with Neutral |

---

## 🚀 Installation

### Prerequisites

- Python 3.8+ (tested on 3.12.9)
- pip (Python package installer)
- Webcam (built-in or external)
- Internet connection (for initial setup)

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/anindita-19/emotion-detection-app.git
cd emotion-detection-app
```

2. **Create virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Verify installation**
```bash
streamlit --version
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
```

5. **Run the application**
```bash
streamlit run web_app.py
```

The app will automatically open in your browser at `http://localhost:8501`

---

## 💻 Usage

### Starting the App

1. Activate your virtual environment
2. Run `streamlit run web_app.py`
3. Click **START** button to activate webcam
4. Allow camera permissions when prompted
5. Position your face in the frame

### Sidebar Controls

- ✅ **Show Bounding Box**: Toggle face detection rectangles
- ✅ **Show Confidence Score**: Display prediction percentages

### Tips for Best Results

- 💡 Ensure good, even lighting
- 💡 Face the camera directly
- 💡 Stay within 2-4 feet from camera
- 💡 Avoid cluttered backgrounds
- 💡 Make clear facial expressions

---

## ⚠️ Limitations

### 1. Significant Overfitting 🔴

**Problem**: Model memorizes training data instead of learning generalizable patterns.

- Training Accuracy: 98.57%
- Validation Accuracy: 61.08%
- **Gap**: ~37.5 percentage points

**Causes**:
- No data augmentation used
- No regularization (dropout, L2)
- Limited dataset size (16,082 images)
- Model may be too complex for dataset

### 2. Emotion-Specific Challenges

**Struggles with**:
- **Angry**: Often misclassified
- **Sad**: Frequently confused with Neutral

**Performs well with**:
- **Happy**: Clear smile detection
- **Neutral**: Consistent recognition
- **Surprised**: Distinctive wide-eyed expressions

### 3. Environmental Sensitivity

- Poor lighting affects face detection
- Harsh shadows distort features
- Low light may prevent face detection entirely

### 4. Limited Scope

- Only 5 emotion categories (missing: Fear, Disgust, etc.)
- Single face optimization (may struggle with multiple faces)
- Trained on static images (video may differ)

---

## 🔧 Future Improvements

### High Priority

1. **Address Overfitting**
   - Implement data augmentation (rotation, flip, brightness)
   - Add Dropout layers (0.3-0.5 rate)
   - Apply L2 regularization
   - Use Early Stopping
   - Implement Batch Normalization

2. **Improve Lighting Robustness**
   - Apply histogram equalization
   - Use CLAHE preprocessing
   - Train with varied lighting conditions

3. **Expand Dataset**
   - Collect more diverse training data
   - Include varied demographics
   - Add challenging conditions (glasses, facial hair)

### Medium Priority

4. **Architecture Improvements**
   - Experiment with transfer learning (VGG16, ResNet, MobileNet)
   - Try different layer configurations
   - Compare optimizers (Adam vs SGD vs RMSprop)

5. **Better Face Detection**
   - Replace Haar Cascade with MTCNN or Dlib
   - Add facial landmark detection

6. **More Emotions**
   - Add Fear, Disgust, Contempt
   - Train on larger public datasets (FER2013, AffectNet)

### Low Priority

7. **UI/UX Enhancements**
   - Emotion history tracking over time
   - Export statistics to CSV
   - Dark/light theme toggle
   - Screenshot/recording functionality

---

## 🛠️ Tech Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.12.9 | Programming language |
| **TensorFlow/Keras** | 2.18.0 | Deep learning framework |
| **OpenCV** | 4.8.1.78 | Computer vision & face detection |
| **NumPy** | 1.26.4 | Numerical computing |
| **Streamlit** | 1.54.0 | Web application framework |
| **streamlit-webrtc** | 0.47.1 | Real-time video streaming |

### Supporting Libraries
- **scikit-learn**: Data preprocessing, train/test split, label encoding
- **pickle**: Data serialization
- **aiortc**: WebRTC implementation
- **av**: Audio/video processing

---

## 📁 Project Structure

```
emotion-detection-app/
│
├── model/
│   └── emotion_detection_model.h5      # Trained model (627K parameters)
│
├── haarcascade/                         # Optional (falls back to OpenCV)
│   └── haarcascade_frontalface_default (1).xml
│
├── web_app.py                           # Main Streamlit application
├── requirements.txt                     # Python dependencies
├── .gitignore                           # Git ignore rules
├── README.md                            # This file
│
└── venv/                                # Virtual environment (not in Git)

NOT INCLUDED (Training files):
├── CTTC_PROJECT.ipynb                   # Model training notebook
├── data/                                # Pickled data (not in repo)
│   ├── images.p
│   └── labels.p
└── Emotion/                             # Raw dataset (not in repo)
    ├── Angry/
    ├── Happy/
    ├── Neutral/
    ├── Sad/
    └── Surprised/
```

**Note**: The training data, pickle files, and raw dataset are NOT included in this repository due to size and licensing constraints.

---

## 🙏 Acknowledgments

- **Dataset**: Provided by course instructor for educational purposes
- **TensorFlow Team**: Deep learning framework
- **OpenCV Team**: Computer vision tools
- **Streamlit Team**: Web application framework
- **Google Colab**: Free GPU/CPU training resources

---

## 📄 License

This project is licensed under the **MIT License** - see below for details.

```
MIT License

Copyright (c) 2026 Anindita

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📧 Contact

- **GitHub**: [@anindita-19](https://github.com/anindita-19)

### Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m "Add feature"`
4. Push to branch: `git push origin feature-name`
5. Open a Pull Request

### Reporting Issues

Found a bug? Please open an issue on GitHub with:
- Clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Screenshots (if applicable)

---

## 🎓 Educational Purpose

This project was developed as part of a machine learning/computer science course to demonstrate:
- Deep Learning fundamentals
- Convolutional Neural Networks
- Computer Vision techniques
- Model deployment with web frameworks

**Academic Integrity**: This project is shared for educational purposes. If using for academic submissions, please follow your institution's academic honesty policies.

---

## 🌟 Support

If you found this project helpful or interesting, please give it a ⭐ on GitHub!

---

<div align="center">

**Made with ❤️ for learning and exploration**

**Last Updated**: February 2026

[⬆ Back to Top](#-real-time-emotion-detection-web-application-)

</div>