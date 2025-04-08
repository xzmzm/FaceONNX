from pydantic import BaseModel
from typing import List, Tuple, Optional

# --- Pydantic Models ---

class RegistrationRequest(BaseModel):
    """Model for the registration request body (used with Form data)."""
    label: str

class FaceDetectionResult(BaseModel):
    """Mirrors C# FaceDetectionResult structure for YOLOv5s-face output."""
    box: List[int] # [x1, y1, x2, y2]
    score: float # Combined objectness and class probability
    landmarks_5pt: List[Tuple[int, int]] # List of 5 (x, y) tuples for eyes, nose, mouth corners

class RecognitionResponse(BaseModel):
    """Model for the recognition API response."""
    label: str
    similarity: float
    query_embedding: Optional[List[float]] = None
    matched_embedding: Optional[List[float]] = None
    matched_image_filename: Optional[str] = None
    query_landmarks_5pt: Optional[List[Tuple[int, int]]] = None # 5-point landmarks from detection
    query_landmarks_68pt: Optional[List[Tuple[int, int]]] = None # Optional 68-point landmarks

class GalleryDataResponse(BaseModel):
    """Model for the gallery data endpoint."""
    # Structure: { "label": [ {"embedding": [float_list], "image_filename": "unique_image.jpg"}, ... ] }
    data: dict[str, list[dict[str, list[float] | str]]]

class RegisteredLabelsResponse(BaseModel):
    """Model for the registered labels endpoint."""
    labels: List[str]