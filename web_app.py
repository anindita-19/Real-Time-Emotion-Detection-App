"""
Emotion Detection Web Application
A real-time emotion detection system using Deep Learning and OpenCV
Built with Streamlit for an interactive web interface
"""

import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

# Page Configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
    <style>
    /* Main background - Dark gradient for better contrast */
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
        background-attachment: fixed;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Custom card styling */
    .stApp {
        background: transparent;
    }
    
    /* Header styling */
    h1 {
        color: #ffffff !important;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
        padding: 20px 0;
        animation: fadeIn 1s ease-in;
    }
    
    h2, h3 {
        color: #ffffff !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.3);
    }
    
    /* Paragraph text */
    p, li, label {
        color: #ffffff !important;
    }
    
    /* Info/Success/Warning boxes */
    .stAlert {
        background-color: rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white !important;
        border-radius: 25px;
        border: none;
        padding: 12px 30px;
        font-weight: bold;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
    }
    
    /* Checkbox styling */
    .stCheckbox {
        color: white !important;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 24px !important;
        font-weight: bold !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #e0e0e0 !important;
    }
    
    /* Video container */
    video {
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        border: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Divider */
    hr {
        border-color: rgba(255, 255, 255, 0.2) !important;
        margin: 30px 0;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Instructions box */
    .instruction-box {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%);
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 25px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Status indicators */
    .status-active {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    </style>
""", unsafe_allow_html=True)


# ========== LOADING FUNCTIONS ==========

@st.cache_resource
def load_cascade():
    """
    Load Haar Cascade for face detection
    First tries the user's custom path, then falls back to OpenCV's built-in cascade
    """
    with st.spinner("🔍 Loading face detection model..."):
        time.sleep(0.5)  # Brief pause for UX
        
        # Try user's custom cascade first
        try:
            face_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default (1).xml")
            if not face_cascade.empty():
                return face_cascade
        except:
            pass
        
        # Fallback to OpenCV's built-in cascade (works everywhere)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if face_cascade.empty():
            st.error("❌ Error: Could not load Haar Cascade classifier!")
            return None
        return face_cascade


@st.cache_resource
def load_emotion_model():
    """
    Load the trained emotion detection model
    Uses Streamlit's cache to load only once
    """
    with st.spinner("🧠 Loading emotion detection model... Please be patient ⏳"):
        time.sleep(1)  # Brief pause for UX
        try:
            # Load model from YOUR path: model/emotion_detection_model.h5
            model = load_model('model/emotion_detection_model.h5')
            return model
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            st.info("💡 Make sure 'emotion_detection_model.h5' is in the 'model' folder!")
            return None


# Emotion labels - UPDATED to match YOUR model (5 emotions)
EMOTION_LABELS = {
    0: "Angry",
    1: "Happy",
    2: "Neutral",
    3: "Sad",
    4: "Surprised"
}

# Emotion emoji mapping for better visualization
EMOTION_EMOJIS = {
    "Angry": "😠",
    "Happy": "😊",
    "Neutral": "😐",
    "Sad": "😢",
    "Surprised": "😲"
}

# Color mapping for emotion labels (BGR format for OpenCV)
EMOTION_COLORS = {
    "Angry": (0, 0, 255),       # Red
    "Happy": (0, 200, 0),       # Green (changed from yellow for better visibility)
    "Neutral": (200, 200, 200), # Light Gray
    "Sad": (255, 100, 0),       # Blue-ish
    "Surprised": (0, 165, 255)  # Orange
}

# Text colors for each emotion (BGR format) - for better readability
TEXT_COLORS = {
    "Angry": (255, 255, 255),    # White text on red
    "Happy": (0, 0, 0),          # Black text on green (fixed visibility!)
    "Neutral": (0, 0, 0),        # Black text on light gray
    "Sad": (255, 255, 255),      # White text on blue
    "Surprised": (0, 0, 0)       # Black text on orange
}


# ========== VIDEO TRANSFORMER CLASS ==========

class EmotionVideoTransformer(VideoTransformerBase):
    """
    Custom video transformer for real-time emotion detection
    Processes each frame from the webcam
    """
    
    def __init__(self):
        self.face_cascade = load_cascade()
        self.emotion_model = load_emotion_model()
        self.show_bbox = True
        self.show_confidence = True
    
    def transform(self, frame):
        """
        Process each video frame - matches YOUR original app.py logic
        """
        img = frame.to_ndarray(format="bgr24")
        
        if self.face_cascade is None or self.emotion_model is None:
            # Display error message on video
            cv2.putText(img, "Model Loading Error!", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return img
        
        # Convert to grayscale for face detection (same as your app.py)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces (using your same parameters)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,  # Same as your app.py
            minNeighbors=5    # Same as your app.py
        )
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face ROI
            face = gray[y:y+h, x:x+w]
            
            # Preprocess exactly like your app.py
            face = cv2.resize(face, (48, 48))
            face = face / 255.0  # Normalize to [0, 1]
            face = face.reshape(1, 48, 48, 1)  # Reshape for model input
            
            # Predict emotion
            try:
                prediction = self.emotion_model.predict(face, verbose=0)
                emotion_index = np.argmax(prediction)
                emotion = EMOTION_LABELS[emotion_index]
                confidence = prediction[0][emotion_index] * 100
                
                # Get color for this emotion
                color = EMOTION_COLORS.get(emotion, (0, 255, 0))
                
                # Draw bounding box
                if self.show_bbox:
                    cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
                
                # Prepare label text
                if self.show_confidence:
                    label = f"{emotion} {EMOTION_EMOJIS.get(emotion, '')} ({confidence:.1f}%)"
                else:
                    label = f"{emotion} {EMOTION_EMOJIS.get(emotion, '')}"
                
                # Calculate text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
                )
                
                # Draw background rectangle for text
                cv2.rectangle(
                    img,
                    (x, y - text_height - 10),
                    (x + text_width + 10, y),
                    color,
                    -1
                )
                
                # Draw text with appropriate color for readability
                text_color = TEXT_COLORS.get(emotion, (255, 255, 255))
                cv2.putText(
                    img, label,
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    text_color,  # Use emotion-specific text color
                    2
                )
                
            except Exception as e:
                # Display error on frame
                cv2.putText(img, f"Prediction Error", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return img


# ========== MAIN APPLICATION ==========

def main():
    """
    Main application function
    """
    
    # Show loading message at startup
    if 'first_load' not in st.session_state:
        st.session_state.first_load = True
        with st.spinner("🚀 App is starting up… Please wait a few seconds."):
            time.sleep(2)
    
    # ========== HEADER ==========
    st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <h1 style='font-size: 3em; margin-bottom: 10px; color: #ffffff; text-shadow: 0 0 20px rgba(139, 92, 246, 0.8), 0 0 40px rgba(99, 102, 241, 0.6);'>
                😊 Emotion Detection 🎭
            </h1>
            <p style='color: #e0e0e0; font-size: 20px; margin-top: 0;'>
                Powered by Deep Learning & Computer Vision
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========== SIDEBAR ==========
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        st.markdown("### 📊 Detection Options")
        show_bbox = st.checkbox("Show Bounding Box", value=True)
        show_confidence = st.checkbox("Show Confidence Score", value=True)
        
        st.markdown("---")
        
        st.markdown("### 📝 About")
        st.info("""
        **Emotion Detection App**
        
        This application uses:
        - 🔍 **Haar Cascade** for face detection
        - 🧠 **Deep Learning CNN** for emotion classification
        - 🎥 **Real-time processing** via webcam
        
        **Detectable Emotions:**
        - 😠 Angry
        - 😊 Happy
        - 😐 Neutral
        - 😢 Sad
        - 😲 Surprised
        """)
        
        st.markdown("---")
        
        st.markdown("### 💡 Tips")
        st.success("""
        - Ensure good lighting
        - Face the camera directly
        - Stay within frame
        - Allow camera permissions
        """)
        
        st.markdown("---")
        
        st.markdown("### 🛠️ Tech Stack")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("![Python](https://img.shields.io/badge/Python-3.12-blue)")
            st.markdown("![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)")
        with col2:
            st.markdown("![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green)")
            st.markdown("![Streamlit](https://img.shields.io/badge/Streamlit-1.54-red)")
    
    # ========== MAIN CONTENT ==========
    
    # Instructions
    st.markdown("## 📹 Live Emotion Detection")
    st.markdown("""
        <div class='instruction-box'>
            <h3 style='color: white; margin-top: 0;'>🎯 Instructions:</h3>
            <ol style='color: white; font-size: 16px; line-height: 1.8;'>
                <li>Click <strong>"START"</strong> button below to activate your webcam</li>
                <li>Allow camera permissions when prompted by your browser</li>
                <li>Position your face in the frame</li>
                <li>The app will detect your face and predict your emotion in real-time!</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)
    
    # WebRTC Configuration
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # Create video streamer
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        webrtc_ctx = webrtc_streamer(
            key="emotion-detection",
            video_transformer_factory=EmotionVideoTransformer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 1280},
                    "height": {"ideal": 720}
                },
                "audio": False
            },
            async_processing=True,
        )
    
    # Update transformer settings
    if webrtc_ctx.video_transformer:
        webrtc_ctx.video_transformer.show_bbox = show_bbox
        webrtc_ctx.video_transformer.show_confidence = show_confidence
    
    # Status indicator
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if webrtc_ctx.state.playing:
            st.success("🟢 Camera Active")
        else:
            st.info("🔴 Camera Inactive")
    
    with col2:
        st.metric("Model Status", "✅ Loaded" if load_emotion_model() is not None else "❌ Error")
    
    with col3:
        st.metric("Face Detector", "✅ Ready" if load_cascade() is not None else "❌ Error")
    
    # ========== FOOTER ==========
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: white; padding: 20px;'>
            <p style='font-size: 14px;'>
                Made with ❤️ using Streamlit | Powered by TensorFlow & OpenCV
            </p>
            <p style='font-size: 12px; opacity: 0.8;'>
                © 2026 Emotion Detection App | All Rights Reserved
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
