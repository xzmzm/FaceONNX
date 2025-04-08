import numpy as np
import cv2
import onnxruntime as ort
from typing import List, Tuple

# Use absolute imports from backend package
from backend.config import EMBEDDING_INPUT_SIZE

def preprocess_image_embedding(cropped_face_np: np.ndarray, input_size: Tuple[int, int] = EMBEDDING_INPUT_SIZE) -> np.ndarray | None:
    """
    Prepares the *cropped* face for the embedding model by resizing and normalizing.
    Matches the C# FaceEmbedder preprocessing: resize, normalize to [-1, 1].
    """
    if cropped_face_np is None or cropped_face_np.size == 0:
        print("Error: Input cropped face is empty for embedding preprocessing.")
        return None

    try:
        # Resize the cropped face to the required input size (e.g., 128x128 for ResNet27)
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
    except Exception as e:
        print(f"Error during embedding preprocessing: {e}")
        return None


def get_embedding(embedding_session: ort.InferenceSession, embedding_blob: np.ndarray) -> List[float] | None:
    """
    Generates the 512-dimension embedding vector using the provided ONNX session.
    Returns the raw embedding vector as a list (no L2 normalization, matching C# FaceEmbedder).
    """
    if embedding_blob is None:
        print("Error: Input blob is None for embedding generation.")
        return None

    try:
        input_name = embedding_session.get_inputs()[0].name
        output_name = embedding_session.get_outputs()[0].name

        embedding_vector = embedding_session.run([output_name], {input_name: embedding_blob})[0]

        # Return the raw embedding vector (as list for JSON compatibility)
        # Remove L2 normalization to match C# FaceEmbedder output
        return embedding_vector.flatten().astype(np.float32).tolist()
    except Exception as e:
        print(f"Error during embedding inference: {e}")
        return None