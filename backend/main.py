import io
import os
import numpy as np
import cv2
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse # Added for serving images
from pydantic import BaseModel
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import math
import json
import uuid # For generating unique filenames
import shutil # For saving UploadFile

# --- Configuration ---
MODEL_DIR = "../netstandard/FaceONNX.Models/models/onnx"
DETECTION_MODEL_PATH = os.path.join(MODEL_DIR, "yolov5s-face.onnx")
EMBEDDING_MODEL_PATH = os.path.join(MODEL_DIR, "recognition_resnet27.onnx")
# LANDMARK_MODEL_PATH no longer needed as we match C# FaceEmbedder pipeline
EMBEDDINGS_FILE = "embeddings.json"
GALLERY_DIR = "gallery_images" # Directory to store registered images

# Create gallery directory if it doesn't exist
os.makedirs(GALLERY_DIR, exist_ok=True)
# Check if models exist
if not all(os.path.exists(p) for p in [DETECTION_MODEL_PATH, EMBEDDING_MODEL_PATH]):
    raise FileNotFoundError("Detection or Embedding ONNX model not found in FaceONNX.Models/models/onnx/. Please ensure the models submodule is initialized.")

# --- ONNX Model Loading ---
# Use CPUExecutionProvider explicitly
providers = ['CPUExecutionProvider']
detection_session = ort.InferenceSession(DETECTION_MODEL_PATH, providers=providers)
# landmark_session removed - not used in C# FaceEmbedder equivalent pipeline
embedding_session = ort.InferenceSession(EMBEDDING_MODEL_PATH, providers=providers)
# --- In-memory storage for embeddings and image filenames ---
# Structure: { "label": [ {"embedding": [float_list], "image_filename": "unique_image.jpg"}, ... ] }
registered_embeddings = {} # Initialize correctly

# --- FastAPI App ---
app = FastAPI(title="Face Recognition API")

# --- CORS Middleware ---
# Allows requests from the frontend development server
origins = [
    "http://localhost:5173", # Default Vite dev server
    "http://127.0.0.1:5173", # Also allow loopback IP
    # Add your production frontend URL here if needed
    # "https://your-frontend-domain.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # List of allowed origins
    allow_credentials=True, # Allow cookies
    allow_methods=["*"],    # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],    # Allow all headers
)
# --- End CORS Middleware ---

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
    query_embedding: list[float] | None = None
    matched_embedding: list[float] | None = None
    matched_image_filename: str | None = None
    query_landmarks: list[tuple[int, int]] | None = None # Landmarks detected on the query image

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
                     # Basic validation of inner structure (check for image_filename)
                     valid_structure = True
                     total_embeddings = 0
                     for label, entries in loaded_data.items():
                         if isinstance(entries, list):
                             total_embeddings += len(entries)
                             for entry in entries: # Check for new structure field
                                 if not (isinstance(entry, dict) and "embedding" in entry and "image_filename" in entry): # Check for image_filename key
                                     valid_structure = False
                                     break
                         else:
                             valid_structure = False
                         if not valid_structure:
                             break

                     if valid_structure:
                         registered_embeddings = loaded_data
                         print(f"Loaded {total_embeddings} embeddings/images for {len(registered_embeddings)} labels from {EMBEDDINGS_FILE}")
                     else:
                         print(f"Warning: Invalid data structure in {EMBEDDINGS_FILE}. Starting with empty embeddings.")
                         registered_embeddings = {}
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
        # Ensure parent directory for embeddings file exists
        if EMBEDDINGS_FILE and os.path.dirname(EMBEDDINGS_FILE):
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

def process_image_to_embedding_and_landmarks(image_bytes: bytes) -> tuple[list | None, list[tuple[int, int]] | None]:
    """Processes an uploaded image to get face embedding and landmarks."""
    try:
        # Read image using OpenCV
        image_np_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image_np_bgr is None:
            print("Error: Could not decode image.")
            return None, None

        # 1. Detect Faces
        detections = detect_faces(image_np_bgr) # Returns list sorted by score
        if not detections:
            print("No face detected.")
            return None, None

        # Select the highest scoring face
        best_detection = detections[0]
        bbox = best_detection["box"] # [x1, y1, x2, y2]
        landmarks_5pt = best_detection["landmarks"] # Ensure landmarks are extracted
        print(f"Detected best face bbox: {bbox} with score {best_detection['score']:.4f}")

        # 2. Crop Face
        x1, y1, x2, y2 = bbox
        # Ensure coordinates are within image bounds before cropping
        h, w = image_np_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        if x1 >= x2 or y1 >= y2:
             print(f"Warning: Invalid crop dimensions derived from bbox {bbox}. Skipping.")
             return None, None

        cropped_face_np = image_np_bgr[y1:y2, x1:x2]

        if cropped_face_np.size == 0:
            print("Warning: Cropped face image is empty.")
            return None, None
        print(f"Cropped face shape: {cropped_face_np.shape}")

        # 3. Preprocess for Embedding (Resize 128x128, Normalize [-1, 1])
        embedding_blob = preprocess_image_embedding(cropped_face_np) # Handles resize and normalization

        # 4. Get Embedding (Raw, no L2 norm)
        if embedding_blob is None:
            print("Could not preprocess for embedding.")
            return None, None # Return None for both if preprocessing fails
        embedding = get_embedding(embedding_blob)
        if embedding is None:
            print("Could not generate embedding.")
            return None, None # Return None for both if embedding fails
        print(f"Generated embedding of length: {len(embedding)}, Landmarks: {landmarks_5pt}")

        return embedding, landmarks_5pt

    except Exception as e:
        import traceback
        print(f"Error processing image: {e}")
        traceback.print_exc()
        return None, None


# --- API Endpoints ---

@app.post("/register")
async def register_face(label: str = Form(...), file: UploadFile = File(...)):
    """Registers a face embedding and saves the image."""
    if not label:
        raise HTTPException(status_code=400, detail="Label cannot be empty.")

    # Ensure filename is safe and generate a unique path
    # Use original extension, default to .jpg if unknown
    file_extension = os.path.splitext(file.filename)[1] if file.filename else '.jpg'
    if not file_extension: file_extension = '.jpg' # Ensure there's an extension
    # Sanitize label for use in filename (replace spaces, etc.) - basic example
    safe_label = "".join(c if c.isalnum() else "_" for c in label)
    unique_filename = f"{safe_label}_{uuid.uuid4().hex[:8]}{file_extension}"
    image_save_path = os.path.join(GALLERY_DIR, unique_filename) # Full path for saving

    # Save the uploaded file
    try:
        # Read the file content first for processing
        image_bytes = await file.read()
        # Save the file using shutil.copyfileobj
        await file.seek(0) # Reset file pointer after reading
        with open(image_save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"Saved image to: {image_save_path}")
    except Exception as e:
        print(f"Error saving uploaded image: {e}")
        raise HTTPException(status_code=500, detail="Could not save uploaded image.")
    finally:
        await file.close() # Ensure file is closed

    # Process image for embedding and landmarks (using the bytes already read)
    embedding, _ = process_image_to_embedding_and_landmarks(image_bytes) # Call renamed function, ignore landmarks

    if embedding is None:
        # Attempt to clean up saved image if embedding failed
        if os.path.exists(image_save_path):
            try:
                os.remove(image_save_path)
                print(f"Removed image {image_save_path} due to embedding failure.")
            except OSError as e:
                print(f"Error removing image {image_save_path}: {e}")
        raise HTTPException(status_code=400, detail="Could not process image or detect face for embedding.")

    # Create the new entry with embedding and image filename
    # Ensure embedding is not None before creating entry
    if embedding is None:
         # This case should have been caught earlier, but double-check
         if os.path.exists(image_save_path):
             try:
                 os.remove(image_save_path)
             except OSError: pass # Ignore error if removal fails
         raise HTTPException(status_code=400, detail="Failed to generate embedding for registration.")

    new_entry = {"embedding": embedding, "image_filename": unique_filename}

    if label in registered_embeddings:
        # Ensure existing entries are lists of dicts
        if not isinstance(registered_embeddings[label], list):
             print(f"Warning: Corrupted data structure for label '{label}', reinitializing.")
             registered_embeddings[label] = []
        # Further check if elements are dicts (optional, basic check done in load)
        registered_embeddings[label].append(new_entry)
    else:
        registered_embeddings[label] = [new_entry] # Start with a list containing the new entry dict

    print(f"Registered '{label}'. Total embeddings for label: {len(registered_embeddings[label])}")
    save_embeddings() # Save after modification
    return {"message": f"Face for '{label}' registered successfully."}

@app.post("/recognize", response_model=RecognitionResponse)
async def recognize_face(file: UploadFile = File(...)):
    """Recognizes a face from an uploaded image against registered faces."""
    if not registered_embeddings:
        raise HTTPException(status_code=400, detail="No faces registered yet.")

    image_bytes = await file.read()
    # Get both embedding and landmarks for the query image
    query_embedding, query_landmarks = process_image_to_embedding_and_landmarks(image_bytes)

    if query_embedding is None:
        # Landmarks might be None or have values, but embedding failed
        raise HTTPException(status_code=400, detail="Could not process query image or detect face for embedding.")

    best_match_label = "unknown"
    highest_similarity = -1.0
    best_matching_entry = None # To store the specific dict {"embedding": ..., "image_filename": ...}

    # Convert query embedding (list) back to numpy array for cosine_similarity
    query_embedding_np = np.array(query_embedding).reshape(1, -1)

    for label, entries_list in registered_embeddings.items():
        if not entries_list: # Skip empty lists for this label
            continue

        # Extract embeddings for this label (assuming new structure)
        embeddings_for_label = [entry["embedding"] for entry in entries_list if isinstance(entry, dict) and "embedding" in entry]
        if not embeddings_for_label:
            continue # Skip if no valid embeddings found for this label

        # Convert list of embedding lists to numpy array for comparison
        embeddings_np = np.array(embeddings_for_label)

        # Calculate similarity against all embeddings for the label
        similarities = cosine_similarity(query_embedding_np, embeddings_np)
        # Get the max similarity for this label
        max_similarity_for_label = np.max(similarities)

        # Find the index of the embedding within this label that gave the max similarity
        best_index_for_label = np.argmax(similarities)
        current_max_similarity = similarities[0, best_index_for_label] # Max similarity for this label

        if current_max_similarity > highest_similarity:
            highest_similarity = current_max_similarity
            best_match_label = label
            # Store the specific entry (dict) that matched best
            best_matching_entry = entries_list[best_index_for_label]

    # You might want a threshold here
    similarity_threshold = 0.5 # Example threshold
    if highest_similarity < similarity_threshold:
         best_match_label = "unknown" # Below threshold
         best_matching_entry = None # No match above threshold

    print(f"Recognition result: Label='{best_match_label}', Similarity={highest_similarity:.4f}")

    # Prepare response data
    matched_embedding_data = best_matching_entry.get("embedding") if best_matching_entry else None
    matched_filename_data = best_matching_entry.get("image_filename") if best_matching_entry else None

    return RecognitionResponse(
        label=best_match_label,
        similarity=float(highest_similarity),
        query_embedding=query_embedding,
        matched_embedding=matched_embedding_data,
        matched_image_filename=matched_filename_data,
        query_landmarks=query_landmarks # Add landmarks to response
    )

@app.get("/registered")
async def get_registered_faces():
    """Returns a list of labels with registered faces."""
    return {"labels": list(registered_embeddings.keys())}

@app.get("/gallery_data")
async def get_gallery_data():
    """Returns the complete dictionary of registered labels and their embeddings."""
    # Ensure embeddings are loaded (though they should be loaded at startup)
    if not registered_embeddings and os.path.exists(EMBEDDINGS_FILE):
        load_embeddings() # Reload if empty but file exists
    # Return the structure {label: [{"embedding": [...], "image_path": "..."}, ...]}
    return registered_embeddings

# --- Endpoint to serve gallery images ---
@app.get("/images/{filename:str}") # Expect filename only
async def get_image(filename: str):
    """Serves an image file from the gallery directory using its filename."""
    # Construct the full path safely
    base_dir = os.path.abspath(GALLERY_DIR)
    requested_path = os.path.abspath(os.path.join(base_dir, filename))

    # Security check: Ensure the resolved path is still within the gallery directory
    # and prevent accessing unintended files like '..'
    if not requested_path.startswith(base_dir) or ".." in filename:
        print(f"Forbidden access attempt: {filename}")
        raise HTTPException(status_code=403, detail="Forbidden: Access denied.")

    if not os.path.isfile(requested_path):
        print(f"Image not found: {filename} (Resolved: {requested_path})")
        raise HTTPException(status_code=404, detail="Image not found.")

    # Use FileResponse to send the image
    return FileResponse(requested_path)


if __name__ == "__main__":
    import uvicorn
    # Make sure the FaceONNX.Models path is correct relative to where you run uvicorn
    print(f"Models expected in: {os.path.abspath(MODEL_DIR)}")
    print(f"Detection model: {os.path.abspath(DETECTION_MODEL_PATH)}")
    # print(f"Landmark model: {os.path.abspath(LANDMARK_MODEL_PATH)}") # Removed
    print(f"Embedding model: {os.path.abspath(EMBEDDING_MODEL_PATH)}")
    print(f"Embeddings file: {os.path.abspath(EMBEDDINGS_FILE)}") # Added print for embeddings file path
    uvicorn.run(app, host="0.0.0.0", port=8000)