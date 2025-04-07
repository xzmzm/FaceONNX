import io
import os
import numpy as np
import cv2
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import math
import json

# --- Configuration ---
MODEL_DIR = "../FaceONNX.Models/models/onnx"
DETECTION_MODEL_PATH = os.path.join(MODEL_DIR, "yolov5s-face.onnx")
EMBEDDING_MODEL_PATH = os.path.join(MODEL_DIR, "recognition_resnet27.onnx")
# LANDMARK_MODEL_PATH no longer needed as we match C# FaceEmbedder pipeline
EMBEDDINGS_FILE = "backend/embeddings.json"

# Check if models exist
# Check if models exist
if not all(os.path.exists(p) for p in [DETECTION_MODEL_PATH, EMBEDDING_MODEL_PATH]):
    raise FileNotFoundError("Detection or Embedding ONNX model not found in FaceONNX.Models/models/onnx/. Please ensure the models submodule is initialized.")

# --- ONNX Model Loading ---
# Use CPUExecutionProvider explicitly
providers = ['CPUExecutionProvider']
detection_session = ort.InferenceSession(DETECTION_MODEL_PATH, providers=providers)
# landmark_session removed - not used in C# FaceEmbedder equivalent pipeline
embedding_session = ort.InferenceSession(EMBEDDING_MODEL_PATH, providers=providers)

# --- In-memory storage for embeddings ---
registered_embeddings = {} # { "label": [list_of_floats] } # Store as lists for JSON

# --- FastAPI App ---
app = FastAPI(title="Face Recognition API")

# --- Pydantic Models ---
class RegistrationRequest(BaseModel):
    label: str

class FaceDetectionResult(BaseModel):
    """Mirrors C# FaceDetectionResult structure"""
    x1: int
    y1: int
    x2: int
    y2: int
    score: float # Combined objectness and class probability
    landmarks: list[tuple[int, int]] # List of 5 (x, y) tuples for eyes, nose, mouth corners

class RecognitionResponse(BaseModel):
    label: str
    similarity: float

# --- Persistence Functions ---

def load_embeddings():
    """Loads registered embeddings from the JSON file."""
    global registered_embeddings
    if os.path.exists(EMBEDDINGS_FILE):
        try:
            with open(EMBEDDINGS_FILE, 'r') as f:
                # Load from JSON, converting lists back to numpy arrays if needed internally
                # For now, keep as lists in registered_embeddings for simplicity with JSON
                loaded_data = json.load(f)
                # Basic validation
                if isinstance(loaded_data, dict):
                     registered_embeddings = loaded_data
                     print(f"Loaded {sum(len(v) for v in registered_embeddings.values())} embeddings for {len(registered_embeddings)} labels from {EMBEDDINGS_FILE}")
                else:
                    print(f"Warning: Invalid format in {EMBEDDINGS_FILE}. Starting with empty embeddings.")
                    registered_embeddings = {}

        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading embeddings from {EMBEDDINGS_FILE}: {e}. Starting with empty embeddings.")
            registered_embeddings = {}
    else:
        print(f"Embeddings file ({EMBEDDINGS_FILE}) not found. Starting with empty embeddings.")
        registered_embeddings = {}

def save_embeddings():
    """Saves the current registered embeddings to the JSON file."""
    try:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(EMBEDDINGS_FILE), exist_ok=True)
        with open(EMBEDDINGS_FILE, 'w') as f:
            json.dump(registered_embeddings, f, indent=4)
        print(f"Saved embeddings to {EMBEDDINGS_FILE}")
    except IOError as e:
        print(f"Error saving embeddings to {EMBEDDINGS_FILE}: {e}")

# Load embeddings at startup
load_embeddings()

# --- Helper Functions ---

# == Face Detection (YOLOv5s-face) ==

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def preprocess_image_detection(image_np_bgr: np.ndarray, input_size=(640, 640)):
    """Prepares image for YOLOv5s-face detection."""
    img, ratio, (dw, dh) = letterbox(image_np_bgr, new_shape=input_size, auto=False, scaleup=False) # Use letterbox for padding/resizing
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB (YOLO expects RGB)
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32) / 255.0  # Normalize to 0.0 - 1.0
    if img.ndim == 3:
        img = np.expand_dims(img, axis=0) # Add batch dimension -> (1, 3, H, W)
    return img, ratio, (dw, dh)

def postprocess_detection(outputs, obj_thres=0.3, iou_thres=0.5, input_shape=(640,640), original_shape=(0,0), ratio=(0,0), pad=(0,0)):
    """
    Postprocesses YOLOv5s-face output.
    Output format: [batch_size, num_boxes, 15]
    Box format: [cx, cy, w, h, obj_conf, lmk1_x, lmk1_y, ..., lmk5_x, lmk5_y]
    Note: This model doesn't have separate class confidence, obj_conf is the primary score.
    """
    # Basic NMS and filtering (replace with more robust NMS if needed)
    outputs = outputs[0] # Get first batch result (shape: num_boxes, 15)
    outputs = outputs[outputs[:, 4] >= obj_thres] # Filter by object confidence (index 4)

    if not outputs.shape[0]:
        return [] # No detections above threshold

    # Convert box format from [cx, cy, w, h] to [x1, y1, x2, y2]
    box = outputs[:, :4]
    box[:, 0] = box[:, 0] - box[:, 2] / 2 # x_center - width/2 = x1
    box[:, 1] = box[:, 1] - box[:, 3] / 2 # y_center - height/2 = y1
    box[:, 2] = box[:, 0] + box[:, 2]     # x1 + width = x2
    box[:, 3] = box[:, 1] + box[:, 3]     # y1 + height = y2
    outputs[:, :4] = box

    # Scale boxes and landmarks back to original image coordinates
    img_h, img_w = original_shape
    # Remove padding
    outputs[:, [0, 2, 5, 7, 9, 11, 13]] -= pad[0]  # x padding (box x1, x2 and landmark x coords)
    outputs[:, [1, 3, 6, 8, 10, 12, 14]] -= pad[1]  # y padding (box y1, y2 and landmark y coords)
    # Rescale
    outputs[:, [0, 2, 5, 7, 9, 11, 13]] /= ratio[0] # width ratio
    outputs[:, [1, 3, 6, 8, 10, 12, 14]] /= ratio[1] # height ratio

    # Clip boxes and landmarks to image boundaries
    outputs[:, [0, 2, 5, 7, 9, 11, 13]] = outputs[:, [0, 2, 5, 7, 9, 11, 13]].clip(0, img_w)  # x coords
    outputs[:, [1, 3, 6, 8, 10, 12, 14]] = outputs[:, [1, 3, 6, 8, 10, 12, 14]].clip(0, img_h)  # y coords

    # Simple IOU based NMS (can be improved)
    keep = []
    scores = outputs[:, 4] # Object confidence score
    order = scores.argsort()[::-1] # Sort by score desc

    boxes = outputs[:, :4]

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / ((boxes[i, 2] - boxes[i, 0] + 1) * (boxes[i, 3] - boxes[i, 1] + 1) + (boxes[order[1:], 2] - boxes[order[1:], 0] + 1) * (boxes[order[1:], 3] - boxes[order[1:], 1] + 1) - inter)

        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]

    final_boxes = outputs[keep]
    # Format results: [x1, y1, x2, y2, score, lmk1_x, lmk1_y, ..., lmk5_x, lmk5_y]
    return final_boxes.tolist()


def detect_faces(image_np_bgr: np.ndarray, obj_thres=0.3, iou_thres=0.5) -> list[dict]:
    """
    Detects all faces in an image using YOLOv5s-face.
    Returns a list of dictionaries, each containing 'box', 'score', and 'landmarks'.
    """
    original_shape = image_np_bgr.shape[:2] # H, W
    input_size = (640, 640) # YOLOv5s input size

    # Preprocess
    input_tensor, ratio, pad = preprocess_image_detection(image_np_bgr, input_size=input_size)

    # Inference
    input_name = detection_session.get_inputs()[0].name
    output_name = detection_session.get_outputs()[0].name
    outputs = detection_session.run([output_name], {input_name: input_tensor})[0]

    # Postprocess to get list of FaceDetectionResult objects
    detections = postprocess_detection(
        outputs,
        obj_thres=obj_thres, # Use objectness threshold
        iou_thres=iou_thres,
        input_shape=input_size,
        original_shape=original_shape,
        ratio=ratio,
        pad=pad
    )

    # Convert raw list [x1, y1, x2, y2, score, lmk1x, lmk1y, ...] to list of dicts
    results = []
    for det in detections:
        landmarks_list = []
        for i in range(5):
            lx = int(round(det[5 + i*2]))
            ly = int(round(det[6 + i*2]))
            landmarks_list.append((lx, ly))

        results.append({
            "box": [int(round(det[0])), int(round(det[1])), int(round(det[2])), int(round(det[3]))], # x1, y1, x2, y2
            "score": float(det[4]),
            "landmarks": landmarks_list
        })

    # Sort by score descending
    results.sort(key=lambda x: x['score'], reverse=True)

    return results # Returns list of detection dicts or empty list

# == Landmark Extraction (PFLD) and Alignment Functions Removed ==
# These are not used in the C# FaceEmbedder equivalent pipeline.
# The pipeline now is: Detect -> Crop -> Resize(128x128) -> Normalize -> Embed

# == Embedding Generation (ResNet27) ==

def preprocess_image_embedding(cropped_face_np: np.ndarray, input_size=(128, 128)):
    """Prepares the *cropped* face for the embedding model by resizing and normalizing."""
    # Ensure input is the correct size (should be from align_face)
    # Resize the cropped face to the required input size (128x128 for ResNet27)
    resized_face = cv2.resize(cropped_face_np, input_size, interpolation=cv2.INTER_LINEAR)

    # Normalize to [-1, 1] and transpose (HWC to NCHW)
    # C# code uses: subtract 127.5, divide by 128. Input is BGR.
    # blobFromImage expects BGR input by default.
    # Use swapRB=False as the C# code doesn't swap channels before normalization.
    input_blob = cv2.dnn.blobFromImage(
        resized_face,
        scalefactor=1.0/128.0,  # C# uses 1/128
        size=input_size,
        mean=(127.5, 127.5, 127.5), # C# subtracts 127.5
        swapRB=False, # Keep BGR order to match C#
        crop=False
    )

    return input_blob

def get_embedding(embedding_blob: np.ndarray):
    """Generates the 512-dimension embedding vector."""
    if embedding_blob is None:
        return None

    input_name = embedding_session.get_inputs()[0].name
    output_name = embedding_session.get_outputs()[0].name

    embedding_vector = embedding_session.run([output_name], {input_name: embedding_blob})[0]

    # Return the raw embedding vector (as list for JSON compatibility)
    # Remove L2 normalization to match C# FaceEmbedder output
    return embedding_vector.flatten().astype(np.float32).tolist()

def process_image_to_embedding(image_bytes: bytes) -> list | None:
    """Processes an uploaded image to get a face embedding."""
    try:
        # Read image using OpenCV
        image_np_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image_np_bgr is None:
            print("Error: Could not decode image.")
            return None

        # 1. Detect Faces
        detections = detect_faces(image_np_bgr) # Returns list sorted by score
        if not detections:
            print("No face detected.")
            return None

        # Select the highest scoring face
        best_detection = detections[0]
        bbox = best_detection["box"] # [x1, y1, x2, y2]
        # landmarks_5pt = best_detection["landmarks"] # Available if needed later
        print(f"Detected best face bbox: {bbox} with score {best_detection['score']:.4f}")

        # 2. Crop Face
        x1, y1, x2, y2 = bbox
        # Ensure coordinates are within image bounds before cropping
        h, w = image_np_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        if x1 >= x2 or y1 >= y2:
             print(f"Warning: Invalid crop dimensions derived from bbox {bbox}. Skipping.")
             return None

        cropped_face_np = image_np_bgr[y1:y2, x1:x2]

        if cropped_face_np.size == 0:
            print("Warning: Cropped face image is empty.")
            return None
        print(f"Cropped face shape: {cropped_face_np.shape}")

        # 3. Preprocess for Embedding (Resize 128x128, Normalize [-1, 1])
        embedding_blob = preprocess_image_embedding(cropped_face_np) # Handles resize and normalization

        # 4. Get Embedding (Raw, no L2 norm)
        if embedding_blob is None:
            print("Could not preprocess for embedding.")
            return None
        embedding = get_embedding(embedding_blob)
        if embedding is None:
            print("Could not generate embedding.")
            return None
        print(f"Generated embedding of length: {len(embedding)}")

        return embedding # Return the final embedding vector

    except Exception as e:
        import traceback
        print(f"Error processing image: {e}")
        traceback.print_exc()
        return None


# --- API Endpoints ---

@app.post("/register")
async def register_face(label: str, file: UploadFile = File(...)):
    """Registers a face embedding from an uploaded image."""
    if not label:
        raise HTTPException(status_code=400, detail="Label cannot be empty.")

    image_bytes = await file.read()
    embedding = process_image_to_embedding(image_bytes)

    if embedding is None:
        raise HTTPException(status_code=400, detail="Could not process image or detect face.")

    # embedding is now a list
    if label in registered_embeddings:
        # Ensure existing entries are lists of lists
        if not isinstance(registered_embeddings[label], list):
             print(f"Warning: Corrupted data for label '{label}', reinitializing.")
             registered_embeddings[label] = []
        registered_embeddings[label].append(embedding)
    else:
        registered_embeddings[label] = [embedding] # Start with a list containing the new list embedding

    print(f"Registered '{label}'. Total embeddings for label: {len(registered_embeddings[label])}")
    save_embeddings() # Save after modification
    return {"message": f"Face for '{label}' registered successfully."}

@app.post("/recognize", response_model=RecognitionResponse)
async def recognize_face(file: UploadFile = File(...)):
    """Recognizes a face from an uploaded image against registered faces."""
    if not registered_embeddings:
        raise HTTPException(status_code=400, detail="No faces registered yet.")

    image_bytes = await file.read()
    query_embedding = process_image_to_embedding(image_bytes)

    if query_embedding is None:
        raise HTTPException(status_code=400, detail="Could not process image or detect face.")

    best_match_label = "unknown"
    highest_similarity = -1.0

    # query_embedding is already a list, convert to numpy array for cosine_similarity

    # Convert query embedding (list) back to numpy array for cosine_similarity
    query_embedding_np = np.array(query_embedding).reshape(1, -1)

    for label, embeddings_list in registered_embeddings.items():
        if not embeddings_list: # Skip empty lists
            continue
        # Convert list of lists back to numpy array for comparison
        embeddings_np = np.array(embeddings_list)
        # Calculate similarity against all embeddings for the label
        similarities = cosine_similarity(query_embedding_np, embeddings_np)
        # Get the max similarity for this label
        max_similarity_for_label = np.max(similarities)

        if max_similarity_for_label > highest_similarity:
            highest_similarity = max_similarity_for_label
            best_match_label = label

    # You might want a threshold here
    similarity_threshold = 0.5 # Example threshold
    if highest_similarity < similarity_threshold:
         best_match_label = "unknown" # Below threshold

    print(f"Recognition result: Label='{best_match_label}', Similarity={highest_similarity:.4f}")
    return RecognitionResponse(label=best_match_label, similarity=float(highest_similarity))

@app.get("/registered")
async def get_registered_faces():
    """Returns a list of labels with registered faces."""
    return {"labels": list(registered_embeddings.keys())}

@app.get("/gallery_data")
async def get_gallery_data():
    """Returns the complete dictionary of registered labels and their embeddings."""
    # Ensure embeddings are loaded (though they should be loaded at startup)
    if not registered_embeddings and os.path.exists(EMBEDDINGS_FILE):
        load_embeddings()
    return registered_embeddings


if __name__ == "__main__":
    import uvicorn
    # Make sure the FaceONNX.Models path is correct relative to where you run uvicorn
    print(f"Models expected in: {os.path.abspath(MODEL_DIR)}")
    print(f"Detection model: {os.path.abspath(DETECTION_MODEL_PATH)}")
    # print(f"Landmark model: {os.path.abspath(LANDMARK_MODEL_PATH)}") # Removed
    print(f"Embedding model: {os.path.abspath(EMBEDDING_MODEL_PATH)}")
    print(f"Embeddings file: {os.path.abspath(EMBEDDINGS_FILE)}") # Added print for embeddings file path
    uvicorn.run(app, host="0.0.0.0", port=8000)