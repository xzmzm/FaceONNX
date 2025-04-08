import os

# --- Configuration ---
# Determine the base directory relative to this config file
# config.py is in backend/, models are in ../netstandard/...
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Goes up one level from backend/
NETSTANDARD_DIR = os.path.join(BASE_DIR, "netstandard")
MODEL_DIR = os.path.join(NETSTANDARD_DIR, "FaceONNX.Models", "models", "onnx")

DETECTION_MODEL_PATH = os.path.join(MODEL_DIR, "yolov5s-face.onnx")
EMBEDDING_MODEL_PATH = os.path.join(MODEL_DIR, "recognition_resnet27.onnx")
LANDMARK_68_MODEL_PATH = os.path.join(MODEL_DIR, "landmarks_68_pfld.onnx")

EMBEDDINGS_FILE = "embeddings.json" # Relative to backend directory where main.py runs
GALLERY_DIR = "gallery_images" # Relative to backend directory

# --- Model Input Sizes ---
DETECTION_INPUT_SIZE = (640, 640)
EMBEDDING_INPUT_SIZE = (128, 128)
LANDMARK_68_INPUT_SIZE = (112, 112) # Input size based on C# Face68LandmarksExtractor analysis

# --- Detection Thresholds (matching C# FaceDetector defaults) ---
DETECTION_OBJ_THRESHOLD = 0.3
DETECTION_CONF_THRESHOLD = 0.4 # Class confidence threshold
DETECTION_NMS_THRESHOLD = 0.5

# --- Recognition Threshold ---
SIMILARITY_THRESHOLD = 0.5

# --- Providers ---
# Use CPUExecutionProvider explicitly, add more if needed (e.g., 'CUDAExecutionProvider')
PROVIDERS = ['CPUExecutionProvider']

# --- Function to check model existence ---
def check_model_files():
    """Checks if essential ONNX model files exist."""
    required_models = [
        DETECTION_MODEL_PATH,
        EMBEDDING_MODEL_PATH,
        LANDMARK_68_MODEL_PATH # Check for landmark model too
    ]
    missing_models = [p for p in required_models if not os.path.exists(p)]
    if missing_models:
        missing_str = "\n - ".join(missing_models)
        raise FileNotFoundError(
            f"Essential ONNX models not found. Please ensure the 'FaceONNX.Models' submodule is initialized and the following files exist:\n - {missing_str}"
        )
    print("All required ONNX model files found.")