import numpy as np
import cv2
import onnxruntime as ort
from typing import List, Dict, Tuple

# Use absolute imports from backend package
from backend.utils import letterbox
from backend.models import FaceDetectionResult
from backend.config import DETECTION_INPUT_SIZE, DETECTION_OBJ_THRESHOLD, DETECTION_CONF_THRESHOLD, DETECTION_NMS_THRESHOLD # Import thresholds

def preprocess_image_detection(image_np_bgr: np.ndarray, input_size: Tuple[int, int] = DETECTION_INPUT_SIZE) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """Prepares image for YOLOv5s-face detection."""
    img, ratio, (dw, dh) = letterbox(image_np_bgr, new_shape=input_size, auto=False, scaleup=False) # Use letterbox for padding/resizing
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB (YOLO expects RGB)
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32) / 255.0  # Normalize to 0.0 - 1.0
    if img.ndim == 3:
        img = np.expand_dims(img, axis=0) # Add batch dimension -> (1, 3, H, W)
    return img, ratio, (dw, dh)

def postprocess_detection(
    outputs: np.ndarray,
    obj_thres: float = DETECTION_OBJ_THRESHOLD, # Use config default
    conf_thres: float = DETECTION_CONF_THRESHOLD, # Add class confidence threshold parameter
    iou_thres: float = DETECTION_NMS_THRESHOLD, # Use config default
    input_shape: Tuple[int, int] = DETECTION_INPUT_SIZE,
    original_shape: Tuple[int, int] = (0,0),
    ratio: Tuple[float, float] = (0.0, 0.0),
    pad: Tuple[float, float] = (0.0, 0.0)
) -> List[FaceDetectionResult]:
    """
    Postprocesses YOLOv5s-face output.
    Output format: [batch_size, num_boxes, 16] (corrected: 1 box + 5 landmarks * 2 coords + 1 score = 15, not 16)
    Box format: [cx, cy, w, h, obj_conf, lmk1_x, lmk1_y, ..., lmk5_x, lmk5_y]
    """
    outputs = outputs[0] # Get first batch result (shape: num_boxes, 15)
    outputs = outputs[outputs[:, 4] >= obj_thres] # Filter by object confidence (index 4)

    # Filter by class confidence (index 5 for YOLOv5s-face, assuming single class 'Face')
    # Match C# logic: prediction[j + yoloSquare] where yoloSquare=15, classes=1 -> index 15? No, C# code is confusing.
    # Let's stick to the most plausible interpretation for yolov5s-face: index 5 is the class score.
    # The C# code checks max(labels) > ConfidenceThreshold where labels = prediction[5:]
    # For single class, this is just prediction[5] > ConfidenceThreshold
    if outputs.shape[1] > 5: # Check if class scores exist (15 columns total: 4 box + 1 obj_conf + 10 landmarks)
        # Index 5 should be the confidence for the first (and only) class 'Face'
        outputs = outputs[outputs[:, 5] >= conf_thres]
    else:
         # This case shouldn't happen if the model is yolov5s-face, but good to have a warning.
         print("Warning: Detection output shape has fewer columns than expected for class scores. Skipping confidence threshold filter.")

    if not outputs.shape[0]:
        return [] # No detections above threshold

    # Convert box format from [cx, cy, w, h] to [x1, y1, x2, y2]
    box = outputs[:, :4].copy() # Use copy to avoid modifying original outputs array directly here
    box[:, 0] = box[:, 0] - box[:, 2] / 2 # x_center - width/2 = x1
    box[:, 1] = box[:, 1] - box[:, 3] / 2 # y_center - height/2 = y1
    box[:, 2] = box[:, 0] + box[:, 2]     # x1 + width = x2
    box[:, 3] = box[:, 1] + box[:, 3]     # y1 + height = y2
    # Don't assign back to outputs[:,:4] yet, do it after NMS

    # Scale boxes and landmarks back to original image coordinates
    img_h, img_w = original_shape
    coords_to_scale = outputs[:, [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]].copy() # Indices for box and landmarks

    # Remove padding
    coords_to_scale[:, [0, 2, 4, 6, 8, 10, 12]] -= pad[0]  # x padding (box cx, w and landmark x coords) -> indices 0, 2, 5, 7, 9, 11, 13 in original outputs
    coords_to_scale[:, [1, 3, 5, 7, 9, 11, 13]] -= pad[1]  # y padding (box cy, h and landmark y coords) -> indices 1, 3, 6, 8, 10, 12, 14 in original outputs

    # Rescale
    coords_to_scale[:, [0, 2, 4, 6, 8, 10, 12]] /= ratio[0] # width ratio
    coords_to_scale[:, [1, 3, 5, 7, 9, 11, 13]] /= ratio[1] # height ratio

    # Update the original outputs array with scaled coordinates
    outputs[:, [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]] = coords_to_scale

    # Convert scaled [cx, cy, w, h] back to [x1, y1, x2, y2] for clipping and NMS
    scaled_boxes = outputs[:, :4].copy()
    scaled_boxes[:, 0] = scaled_boxes[:, 0] - scaled_boxes[:, 2] / 2 # x1
    scaled_boxes[:, 1] = scaled_boxes[:, 1] - scaled_boxes[:, 3] / 2 # y1
    scaled_boxes[:, 2] = scaled_boxes[:, 0] + scaled_boxes[:, 2]     # x2
    scaled_boxes[:, 3] = scaled_boxes[:, 1] + scaled_boxes[:, 3]     # y2

    # Clip boxes and landmarks to image boundaries
    # Clip boxes (x1, y1, x2, y2)
    scaled_boxes[:, [0, 2]] = scaled_boxes[:, [0, 2]].clip(0, img_w)
    scaled_boxes[:, [1, 3]] = scaled_boxes[:, [1, 3]].clip(0, img_h)
    # Clip landmarks (lx1, ly1, lx2, ly2, ...)
    outputs[:, [5, 7, 9, 11, 13]] = outputs[:, [5, 7, 9, 11, 13]].clip(0, img_w)  # x coords
    outputs[:, [6, 8, 10, 12, 14]] = outputs[:, [6, 8, 10, 12, 14]].clip(0, img_h)  # y coords

    # Simple IOU based NMS (can be improved)
    keep = []
    scores = outputs[:, 4] # Object confidence score
    order = scores.argsort()[::-1] # Sort by score desc

    # Use the scaled_boxes for NMS calculation
    nms_boxes = scaled_boxes

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1: break # Exit if only one box left

        xx1 = np.maximum(nms_boxes[i, 0], nms_boxes[order[1:], 0])
        yy1 = np.maximum(nms_boxes[i, 1], nms_boxes[order[1:], 1])
        xx2 = np.minimum(nms_boxes[i, 2], nms_boxes[order[1:], 2])
        yy2 = np.minimum(nms_boxes[i, 3], nms_boxes[order[1:], 3])

        w = np.maximum(0.0, xx2 - xx1 + 1e-5) # Add epsilon for stability
        h = np.maximum(0.0, yy2 - yy1 + 1e-5)
        inter = w * h

        # Calculate areas using the boxes used for NMS
        area_i = (nms_boxes[i, 2] - nms_boxes[i, 0]) * (nms_boxes[i, 3] - nms_boxes[i, 1])
        area_others = (nms_boxes[order[1:], 2] - nms_boxes[order[1:], 0]) * (nms_boxes[order[1:], 3] - nms_boxes[order[1:], 1])

        ovr = inter / (area_i + area_others - inter + 1e-5) # Add epsilon

        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]

    final_detections_raw = outputs[keep]
    final_boxes_scaled = scaled_boxes[keep] # Get the corresponding scaled boxes

    # Format results into FaceDetectionResult objects
    results = []
    for i, det in enumerate(final_detections_raw):
        landmarks_list = []
        for lm_idx in range(5):
            lx = int(round(det[5 + lm_idx*2]))
            ly = int(round(det[6 + lm_idx*2]))
            landmarks_list.append((lx, ly))

        # Use the final scaled and clipped box coordinates
        box_coords = [int(round(c)) for c in final_boxes_scaled[i]]

        results.append(FaceDetectionResult(
            box=box_coords, # x1, y1, x2, y2
            score=float(det[4]),
            landmarks_5pt=landmarks_list
        ))

    # Sort by score descending (already sorted by NMS, but good practice)
    results.sort(key=lambda x: x.score, reverse=True)

    return results


def detect_faces(
    detection_session: ort.InferenceSession,
    image_np_bgr: np.ndarray,
    # Pass thresholds explicitly to postprocess_detection
    obj_thres: float = DETECTION_OBJ_THRESHOLD,
    conf_thres: float = DETECTION_CONF_THRESHOLD,
    iou_thres: float = DETECTION_NMS_THRESHOLD
) -> List[FaceDetectionResult]:
    """
    Detects all faces in an image using the provided YOLOv5s-face ONNX session.
    Returns a list of FaceDetectionResult objects, sorted by score.
    """
    if image_np_bgr is None:
        print("Error: Input image is None for face detection.")
        return []

    original_shape = image_np_bgr.shape[:2] # H, W

    # Preprocess
    input_tensor, ratio, pad = preprocess_image_detection(image_np_bgr, input_size=DETECTION_INPUT_SIZE)

    # Inference
    try:
        input_name = detection_session.get_inputs()[0].name
        output_name = detection_session.get_outputs()[0].name
        outputs = detection_session.run([output_name], {input_name: input_tensor})[0]
    except Exception as e:
        print(f"Error during detection inference: {e}")
        return []

    # Postprocess to get list of FaceDetectionResult objects
    # Pass all thresholds to the postprocessing function
    detections = postprocess_detection(
        outputs,
        obj_thres=obj_thres,
        conf_thres=conf_thres, # Pass confidence threshold
        iou_thres=iou_thres,
        input_shape=DETECTION_INPUT_SIZE,
        original_shape=original_shape,
        ratio=ratio,
        pad=pad
    )

    return detections