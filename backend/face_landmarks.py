import numpy as np
import cv2
import onnxruntime as ort
from typing import List, Tuple, Optional

# Define input size based on C# code analysis
LANDMARK_68_INPUT_SIZE = (112, 112)

def preprocess_image_landmarks_68(cropped_face_np: np.ndarray, input_size: Tuple[int, int] = LANDMARK_68_INPUT_SIZE) -> Optional[np.ndarray]:
    """
    Prepares the *cropped* face image for the PFLD 68-point landmark model.
    - Resizes to input_size (e.g., 112x112).
    - Normalizes pixel values to [0, 1].
    - Transposes to NCHW format.
    """
    if cropped_face_np is None or cropped_face_np.size == 0:
        print("Error: Input cropped face is empty for landmark preprocessing.")
        return None

    try:
        # Resize to the required input size (112x112 for PFLD in C# example)
        resized_face = cv2.resize(cropped_face_np, input_size, interpolation=cv2.INTER_LINEAR)

        # Convert to float32 and normalize to [0, 1]
        input_blob = resized_face.astype(np.float32) / 255.0

        # Transpose from HWC (Height, Width, Channels) to NCHW (Batch, Channels, Height, Width)
        input_blob = input_blob.transpose(2, 0, 1)
        input_blob = np.expand_dims(input_blob, axis=0) # Add batch dimension
        input_blob = np.ascontiguousarray(input_blob) # Ensure contiguous memory layout

        return input_blob
    except Exception as e:
        print(f"Error during 68-point landmark preprocessing: {e}")
        return None


def extract_landmarks_68(
    landmark_session: ort.InferenceSession,
    landmark_blob: np.ndarray,
    original_crop_width: int,
    original_crop_height: int
) -> Optional[List[Tuple[int, int]]]:
    """
    Extracts 68 face landmarks using the provided ONNX session.
    - Runs inference on the preprocessed blob.
    - Scales the normalized output landmarks back to the original cropped face dimensions.
    """
    if landmark_blob is None:
        print("Error: Input blob is None for 68-point landmark extraction.")
        return None
    if original_crop_width <= 0 or original_crop_height <= 0:
        print(f"Error: Invalid original crop dimensions ({original_crop_width}x{original_crop_height}) for landmark scaling.")
        return None

    try:
        input_name = landmark_session.get_inputs()[0].name
        # Assuming the landmarks are the first output based on typical PFLD models
        # Verify this if the model has multiple outputs
        output_name = landmark_session.get_outputs()[0].name

        # Run inference
        outputs = landmark_session.run([output_name], {input_name: landmark_blob})
        # Output shape is expected to be (1, 136) for 68 points (x, y)
        landmarks_norm = outputs[0].flatten() # Flatten to a 1D array of 136 values

        if landmarks_norm.shape[0] != 136:
            print(f"Error: Unexpected landmark output shape: {landmarks_norm.shape}. Expected 136 values.")
            return None

        # Reshape to (68, 2) and scale back to original cropped image coordinates
        landmarks_scaled = []
        for i in range(0, 136, 2):
            # Landmarks are normalized to the input size (112x112), scale them to the *original crop* size
            x_norm = landmarks_norm[i]
            y_norm = landmarks_norm[i+1]
            # Scale normalized coords (0-1 range) by the dimensions of the face *before* 112x112 resize
            x_scaled = int(round(x_norm * original_crop_width))
            y_scaled = int(round(y_norm * original_crop_height))
            landmarks_scaled.append((x_scaled, y_scaled))

        return landmarks_scaled

    except Exception as e:
        print(f"Error during 68-point landmark inference or postprocessing: {e}")
        import traceback
        traceback.print_exc()
        return None