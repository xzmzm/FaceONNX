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

def align_face_csharp_logic(
    full_image_np: np.ndarray,
    original_box: Tuple[int, int, int, int],
    angle_deg: float,
    scale_factor: float = 1.2, # Factor to scale the initial crop box
    clamp: bool = True # Mimic C# clamp behavior
) -> Optional[np.ndarray]:
    """
    Aligns a face using a process similar to the C# FaceProcessingExtensions.Align:
    1. Scale the original bounding box.
    2. Crop the face using the scaled box from the full image.
    3. Rotate the scaled crop.
    4. Crop the final face from the rotated image using the original box dimensions
       relative to the scaled crop.
    """
    if full_image_np is None or original_box is None or angle_deg is None:
        print("Error: Missing input for align_face_csharp_logic.")
        return None

    img_h, img_w = full_image_np.shape[:2]
    ox1, oy1, ox2, oy2 = original_box
    orig_w = ox2 - ox1
    orig_h = oy2 - oy1

    if orig_w <= 0 or orig_h <= 0:
        print(f"Error: Invalid original box dimensions {original_box}.")
        return None

    # 1. Scale the original bounding box
    center_x = ox1 + orig_w / 2
    center_y = oy1 + orig_h / 2
    scaled_w = orig_w * scale_factor
    scaled_h = orig_h * scale_factor
    sx1 = int(round(center_x - scaled_w / 2))
    sy1 = int(round(center_y - scaled_h / 2))
    sx2 = int(round(center_x + scaled_w / 2))
    sy2 = int(round(center_y + scaled_h / 2))
    scaled_box_w = sx2 - sx1
    scaled_box_h = sy2 - sy1

    # Clamp scaled box coordinates to image boundaries
    csx1, csy1 = max(0, sx1), max(0, sy1)
    csx2, csy2 = min(img_w, sx2), min(img_h, sy2)

    if csx1 >= csx2 or csy1 >= csy2:
        print(f"Warning: Scaled box {sx1, sy1, sx2, sy2} resulted in invalid clamp {csx1, csy1, csx2, csy2}. Skipping alignment.")
        # Fallback: return simple crop using original box? Or None? Let's return None.
        return None # Indicate alignment failure

    # 2. Crop the face using the (clamped) scaled box
    scaled_crop = full_image_np[csy1:csy2, csx1:csx2]
    if scaled_crop.size == 0:
        print("Warning: Scaled crop resulted in empty image. Skipping alignment.")
        return None

    # 3. Rotate the scaled crop
    try:
        (sc_h, sc_w) = scaled_crop.shape[:2]
        center_sc = (sc_w // 2, sc_h // 2)
        # Use negative angle to match the C# Rotate(-angle) call.
        # OpenCV warpAffine rotates counter-clockwise for positive angles.
        # Our calculated angle_deg is positive for counter-clockwise rotation needed.
        # Therefore, negating it aligns with the C# code's explicit negation.
        M = cv2.getRotationMatrix2D(center_sc, -angle_deg, 1.0)
        rotated_scaled_crop = cv2.warpAffine(scaled_crop, M, (sc_w, sc_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    except Exception as e:
        print(f"Error during scaled crop rotation: {e}")
        return None # Indicate alignment failure

    # 4. Calculate final crop coordinates relative to the rotated scaled crop (Simpler method, closer to C# structure)
    #    These are based on the *original* box dimensions, offset by the scaled box's origin *within the full image*.
    final_crop_x1_rel = ox1 - sx1 # Relative start X within the scaled crop (before rotation)
    final_crop_y1_rel = oy1 - sy1 # Relative start Y within the scaled crop (before rotation)

    # Ensure final relative coords are integers and define width/height
    fx1 = int(round(final_crop_x1_rel))
    fy1 = int(round(final_crop_y1_rel))
    fw = int(round(orig_w)) # Final width is original width
    fh = int(round(orig_h)) # Final height is original height
    fx2 = fx1 + fw
    fy2 = fy1 + fh

    # Clamp final crop coordinates to the *rotated scaled crop* boundaries
    rot_h, rot_w = rotated_scaled_crop.shape[:2]
    cfx1, cfy1 = max(0, fx1), max(0, fy1)
    cfx2, cfy2 = min(rot_w, fx2), min(rot_h, fy2)

    if cfx1 >= cfx2 or cfy1 >= cfy2:
         print(f"Warning: Final relative box {fx1, fy1, fx2, fy2} resulted in invalid clamp {cfx1, cfy1, cfx2, cfy2} on rotated crop. Skipping final crop.")
         # Return the rotated scaled crop? Or None? Let's return None.
         return None

    # 5. Crop the final face from the rotated image
    final_aligned_face = rotated_scaled_crop[cfy1:cfy2, cfx1:cfx2]

    if final_aligned_face.size == 0:
        print("Warning: Final aligned crop is empty.")
        return None

    return final_aligned_face


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

        # 2. Preprocess for Landmarks (includes cropping from full image) & Get Crop
        # This step now handles the cropping based on the detection box.
        landmark_blob = None
        cropped_face_np = None # Will be populated by preprocess function
        preprocess_result = None
        if landmark_session is not None:
            print("  Preprocessing for 68-point landmarks (includes cropping)...")
            preprocess_result = preprocess_image_landmarks_68(image_np_bgr, detection.box)
            if preprocess_result:
                landmark_blob, cropped_face_np = preprocess_result
                if cropped_face_np is not None:
                     print(f"  Cropped face shape for landmarks: {cropped_face_np.shape}")
                else:
                     print("  Cropping failed during landmark preprocessing.")
            else:
                print("  Landmark preprocessing failed.")
        else:
             print("  Warning: Landmark session not available, cannot perform alignment. Attempting simple crop for embedding.")
             # Fallback: Perform a simple crop if landmarks/alignment aren't possible/needed
             cropped_face_np = crop_face(image_np_bgr, detection)

        # If cropping failed (either via landmark prep or fallback), skip this face
        if cropped_face_np is None:
            print(f"  Failed to obtain cropped face for face #{i+1}. Skipping.")
            all_processed_results.append(ProcessedFaceResult(detection=detection)) # Add detection result only
            continue

        crop_h, crop_w = cropped_face_np.shape[:2] # Get dimensions from the actual crop obtained

        # 3. Extract 68-point Landmarks (needed for alignment *before* embedding)
        # We always need to try extracting landmarks if we want to align like the C# example
        # 3. Extract 68-point Landmarks (using the blob from step 2)
        landmarks_68pt_raw = None # Relative to the crop
        current_landmarks_68pt_adjusted = None # Adjusted to full image coords
        if landmark_blob is not None and landmark_session is not None:
             print("  Extracting 68-point landmarks from preprocessed blob...")
             landmarks_68pt_raw = extract_landmarks_68(landmark_session, landmark_blob, crop_w, crop_h)
             if landmarks_68pt_raw:
                 # Adjust landmarks to be relative to the full image origin
                 x1_box, y1_box = detection.box[0], detection.box[1]
                 current_landmarks_68pt_adjusted = [(pt[0] + x1_box, pt[1] + y1_box) for pt in landmarks_68pt_raw]
                 print(f"  Extracted {len(landmarks_68pt_raw)} raw 68-point landmarks (relative to crop).")
             else:
                 print("  Could not extract 68-point landmarks from blob.")
        elif landmark_session is not None and landmark_blob is None:
             print("  Landmark blob not available, skipping landmark extraction.")
        # If landmark_session was None, message printed in step 2

        # 4. Calculate Rotation Angle and Align Face
        aligned_face_np = cropped_face_np # Default to unaligned crop
        if landmarks_68pt_raw: # Only align if landmarks were found
            angle = calculate_rotation_angle(landmarks_68pt_raw)
            if angle is not None:
                print(f"  Calculated rotation angle: {angle:.2f} degrees.")
                # Align using the new function, passing the full image and original box
                aligned_face_np = align_face_csharp_logic(image_np_bgr, detection.box, angle)
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
            landmarks_68pt=current_landmarks_68pt_adjusted if extract_68_landmarks else None # Only return if requested by API call
        ))
        # --- End loop for processing each face ---

    return all_processed_results # Return the list of results