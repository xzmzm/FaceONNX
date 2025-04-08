import numpy as np
import cv2
import onnxruntime as ort
from typing import Tuple, List, Optional
import math # Import math for atan2 and degrees

# Use absolute imports from backend package
from backend.face_detection import detect_faces
from backend.face_embedding import preprocess_image_embedding, get_embedding
from backend.face_landmarks import preprocess_image_landmarks_68, extract_landmarks_68
from backend.models import FaceDetectionResult

def crop_face(image_np_bgr: np.ndarray, detection_result: FaceDetectionResult) -> Optional[np.ndarray]:
    """Crops the face from the image based on the detection bounding box."""
    if image_np_bgr is None or detection_result is None:
        return None

    x1, y1, x2, y2 = detection_result.box
    h, w = image_np_bgr.shape[:2]

    # Ensure coordinates are within image bounds before cropping
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    if x1 >= x2 or y1 >= y2:
        print(f"Warning: Invalid crop dimensions derived from bbox {detection_result.box}. Skipping crop.")
        return None

    cropped_face = image_np_bgr[y1:y2, x1:x2]

    if cropped_face.size == 0:
        print("Warning: Cropped face image is empty.")
        return None

    return cropped_face

def calculate_rotation_angle(landmarks_68pt: List[Tuple[int, int]]) -> Optional[float]:
   """Calculates the rotation angle based on eye landmarks to align the face horizontally."""
   if not landmarks_68pt or len(landmarks_68pt) != 68:
       return None

   try:
       # Use standard iBUG 300-W indices for eye corners
       # Left eye: outer corner (36), inner corner (39)
       # Right eye: inner corner (42), outer corner (45)
       left_eye_center_x = (landmarks_68pt[36][0] + landmarks_68pt[39][0]) / 2
       left_eye_center_y = (landmarks_68pt[36][1] + landmarks_68pt[39][1]) / 2
       right_eye_center_x = (landmarks_68pt[42][0] + landmarks_68pt[45][0]) / 2
       right_eye_center_y = (landmarks_68pt[42][1] + landmarks_68pt[45][1]) / 2

       dy = right_eye_center_y - left_eye_center_y
       dx = right_eye_center_x - left_eye_center_x

       angle_rad = math.atan2(dy, dx)
       angle_deg = math.degrees(angle_rad)
       return angle_deg
   except IndexError:
       print("Error: Landmark indices out of bounds during angle calculation.")
       return None
   except Exception as e:
       print(f"Error calculating rotation angle: {e}")
       return None

def align_face_by_rotation(image_np: np.ndarray, angle_deg: float) -> Optional[np.ndarray]:
   """Rotates an image (presumably a face crop) to align eyes horizontally."""
   if image_np is None or angle_deg is None:
       return image_np # Return original if no rotation needed or possible

   try:
       (h, w) = image_np.shape[:2]
       center = (w // 2, h // 2)

       # Get rotation matrix
       # Use negative angle because cv2 rotates counter-clockwise for positive angles
       M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

       # Perform the rotation
       # Use black border color to match C# padding assumption
       aligned_face = cv2.warpAffine(image_np, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
       return aligned_face
   except Exception as e:
       print(f"Error during face alignment rotation: {e}")
       return image_np # Return original on error


# Define a structure to hold results for a single detected face
class ProcessedFaceResult:
    def __init__(self,
                 detection: FaceDetectionResult,
                 embedding: Optional[List[float]] = None,
                 landmarks_68pt: Optional[List[Tuple[int, int]]] = None):
        self.detection = detection # Includes bbox and 5pt landmarks
        self.embedding = embedding
        self.landmarks_68pt = landmarks_68pt

def process_image_full(
    image_np_bgr: np.ndarray,
    detection_session: ort.InferenceSession,
    embedding_session: ort.InferenceSession,
    landmark_session: Optional[ort.InferenceSession] = None, # Optional landmark session
    extract_68_landmarks: bool = False # Flag to control 68 landmark extraction
) -> List[ProcessedFaceResult]: # Return type changed
    """
    Processes an image to find ALL faces, extract their embeddings,
    and optionally extract 5-point and 68-point landmarks for EACH face.

    Args:
        image_np_bgr: The input image as a BGR numpy array.
        detection_session: ONNX session for face detection.
        embedding_session: ONNX session for face embedding.
        landmark_session: Optional ONNX session for 68-point landmarks.
        extract_68_landmarks: If True, attempts to extract 68-point landmarks for each face.

    Returns:
        A list of ProcessedFaceResult objects, one for each successfully processed face.
        Each result contains the original detection info (bbox, 5pt landmarks) and
        potentially the embedding and 68pt landmarks.
        Returns an empty list if no faces are detected.
    """
    all_processed_results: List[ProcessedFaceResult] = []

    if image_np_bgr is None:
        print("Error: Input image is None for processing.")
        return all_processed_results

    # 1. Detect ALL Faces
    detections = detect_faces(detection_session, image_np_bgr) # Returns list sorted by score
    if not detections:
        print("No faces detected.")
        return all_processed_results

    print(f"Detected {len(detections)} faces.")

    # --- Process EACH detected face ---
    for i, detection in enumerate(detections):
        print(f"\nProcessing detected face #{i+1} (Score: {detection.score:.4f})")
        current_embedding: Optional[List[float]] = None
        current_landmarks_68pt: Optional[List[Tuple[int, int]]] = None

        # 2. Crop Face for this detection
        cropped_face_np = crop_face(image_np_bgr, detection)
        if cropped_face_np is None:
            print(f"  Failed to crop face #{i+1}. Skipping embedding/landmarks for this face.")
            # Still add the detection result itself, but without embedding/landmarks
            all_processed_results.append(ProcessedFaceResult(detection=detection))
            continue # Move to the next detection

        crop_h, crop_w = cropped_face_np.shape[:2]
        print(f"  Cropped face shape: {cropped_face_np.shape}")

        # 3. Extract 68-point Landmarks (needed for alignment *before* embedding)
        # We always need to try extracting landmarks if we want to align like the C# example
        landmarks_68pt_raw = None
        current_landmarks_68pt_adjusted = None # Adjusted to full image coords
        if landmark_session is not None and cropped_face_np is not None:
            print("  Extracting 68-point landmarks for alignment...")
            landmark_blob = preprocess_image_landmarks_68(cropped_face_np)
            if landmark_blob is not None:
                landmarks_68pt_raw = extract_landmarks_68(landmark_session, landmark_blob, crop_w, crop_h)
                if landmarks_68pt_raw:
                    x1_crop, y1_crop = detection.box[0], detection.box[1]
                    current_landmarks_68pt_adjusted = [(pt[0] + x1_crop, pt[1] + y1_crop) for pt in landmarks_68pt_raw]
                    print(f"  Extracted {len(current_landmarks_68pt_adjusted)} 68-point landmarks.")
                else:
                    print("  Could not extract 68-point landmarks.")
            else:
                print("  Could not preprocess for 68-point landmarks.")
        elif landmark_session is None:
             print("  Warning: Landmark session not available, cannot perform alignment.")

        # 4. Calculate Rotation Angle and Align Face
        aligned_face_np = cropped_face_np # Default to unaligned crop
        if landmarks_68pt_raw: # Only align if landmarks were found
            angle = calculate_rotation_angle(landmarks_68pt_raw)
            if angle is not None:
                print(f"  Calculated rotation angle: {angle:.2f} degrees.")
                aligned_face_np = align_face_by_rotation(cropped_face_np, angle)
                if aligned_face_np is None: # Handle alignment error
                    print("  Face alignment failed, using original crop.")
                    aligned_face_np = cropped_face_np
                else:
                    print("  Face alignment applied.")
            else:
                print("  Could not calculate rotation angle, using original crop.")
        else:
            print("  No 68pt landmarks found, skipping alignment.")


        # 5. Preprocess and Get Embedding (using the potentially aligned face)
        if aligned_face_np is not None:
            embedding_blob = preprocess_image_embedding(aligned_face_np)
            if embedding_blob is not None:
                current_embedding = get_embedding(embedding_session, embedding_blob)
                if current_embedding is None:
                    print("  Could not generate embedding from aligned face.")
                else:
                    print(f"  Generated embedding of length: {len(current_embedding)} from aligned face.")
            else:
                print("  Could not preprocess aligned face for embedding.")
        else:
             print("  Aligned face is None, cannot generate embedding.")


        # Add the result for this face
        # Store the adjusted 68 landmarks if they were calculated
        all_processed_results.append(ProcessedFaceResult(
            detection=detection,
            embedding=current_embedding,
            landmarks_68pt=current_landmarks_68pt_adjusted if extract_landmarks else None # Only return if requested by API call
        ))
        # --- End loop for processing each face ---

    return all_processed_results # Return the list of results