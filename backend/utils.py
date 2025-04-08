import os
import uuid
import shutil
import numpy as np
import cv2
from fastapi import UploadFile, HTTPException

def letterbox(im, new_shape=(640, 640), color=(0, 0, 0), auto=True, scaleFill=False, scaleup=True, stride=32): # Changed padding color to black (0,0,0)
    """
    Resizes and pads image while meeting stride-multiple constraints.
    Source: YOLOv5 implementation.
    """
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


async def save_uploaded_image(file: UploadFile, label: str, gallery_dir: str) -> tuple[str, bytes]:
    """
    Saves the uploaded image file to the gallery directory with a unique name.
    Returns the unique filename and the image bytes.
    Raises HTTPException on failure.
    """
    # Ensure filename is safe and generate a unique path
    file_extension = os.path.splitext(file.filename)[1] if file.filename else '.jpg'
    if not file_extension: file_extension = '.jpg' # Ensure there's an extension

    # Sanitize label for use in filename (replace spaces, etc.) - basic example
    safe_label = "".join(c if c.isalnum() else "_" for c in label)
    unique_filename = f"{safe_label}_{uuid.uuid4().hex[:8]}{file_extension}"
    image_save_path = os.path.join(gallery_dir, unique_filename) # Full path for saving

    try:
        # Read the file content first for processing later
        image_bytes = await file.read()
        # Save the file using shutil.copyfileobj
        await file.seek(0) # Reset file pointer after reading
        with open(image_save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"Saved image to: {image_save_path}")
        return unique_filename, image_bytes
    except Exception as e:
        print(f"Error saving uploaded image: {e}")
        # Attempt to clean up if partially saved
        if os.path.exists(image_save_path):
            try:
                os.remove(image_save_path)
            except OSError:
                pass # Ignore cleanup error
        raise HTTPException(status_code=500, detail=f"Could not save uploaded image: {e}")
    finally:
        await file.close() # Ensure file is closed

def clean_up_image(image_path: str):
    """Removes an image file if it exists."""
    if os.path.exists(image_path):
        try:
            os.remove(image_path)
            print(f"Removed image {os.path.basename(image_path)} due to processing failure.")
        except OSError as e:
            print(f"Error removing image {os.path.basename(image_path)}: {e}")

def decode_image(image_bytes: bytes) -> np.ndarray | None:
    """Decodes image bytes into an OpenCV BGR numpy array."""
    try:
        image_np_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image_np_bgr is None:
            print("Error: Could not decode image bytes.")
            return None
        return image_np_bgr
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None