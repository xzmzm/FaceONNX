import os
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sklearn.metrics.pairwise import cosine_similarity
import traceback # For detailed error logging

import sys # Add sys import
import os # Add os import

# Ensure the parent directory (FaceONNX) is in the Python path
# This allows running 'python main.py' from the 'backend' directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import modules using absolute imports from the 'backend' package
from backend import config
from backend import models
from backend import utils
from backend import persistence
from backend import face_detection
from backend import face_embedding
from backend import face_landmarks
from backend import processing # Make sure processing is imported

# --- Initialization ---

# Check if model files exist before proceeding
try:
    config.check_model_files()
except FileNotFoundError as e:
    print(f"Error: {e}")
    # Exit or raise a more specific startup error if running in a managed environment
    exit(1) # Exit if models are missing

# Create gallery directory if it doesn't exist
os.makedirs(config.GALLERY_DIR, exist_ok=True)

# Load ONNX models
try:
    print(f"Loading detection model from: {config.DETECTION_MODEL_PATH}")
    detection_session = ort.InferenceSession(config.DETECTION_MODEL_PATH, providers=config.PROVIDERS)
    print(f"Loading embedding model from: {config.EMBEDDING_MODEL_PATH}")
    embedding_session = ort.InferenceSession(config.EMBEDDING_MODEL_PATH, providers=config.PROVIDERS)
    print(f"Loading 68-landmark model from: {config.LANDMARK_68_MODEL_PATH}")
    landmark_session = ort.InferenceSession(config.LANDMARK_68_MODEL_PATH, providers=config.PROVIDERS)
    print("ONNX models loaded successfully.")
except Exception as e:
    print(f"Fatal Error: Could not load ONNX models. {e}")
    traceback.print_exc()
    exit(1)

# Load registered embeddings
registered_embeddings = persistence.load_embeddings(config.EMBEDDINGS_FILE)

# --- FastAPI App Setup ---
app = FastAPI(title="Face Recognition API")

# CORS Middleware
origins = [
    "http://localhost:5173", # Default Vite dev server
    "http://127.0.0.1:5173", # Also allow loopback IP
    # Add production frontend URL if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---

@app.post("/register")
async def register_face(label: str = Form(...), file: UploadFile = File(...)):
    """Registers a face embedding and saves the image."""
    if not label:
        raise HTTPException(status_code=400, detail="Label cannot be empty.")

    image_save_path = None # Initialize to None
    try:
        # Save uploaded image first
        unique_filename, image_bytes = await utils.save_uploaded_image(file, label, config.GALLERY_DIR)
        image_save_path = os.path.join(config.GALLERY_DIR, unique_filename) # Full path

        # Decode image bytes
        image_np_bgr = utils.decode_image(image_bytes)
        if image_np_bgr is None:
             raise HTTPException(status_code=400, detail="Could not decode uploaded image.")

        # Process image for embedding (only need the first/best face for registration)
        all_results = processing.process_image_full(
            image_np_bgr=image_np_bgr,
            detection_session=detection_session,
            embedding_session=embedding_session,
            landmark_session=landmark_session, # ALWAYS pass session for alignment
            extract_68_landmarks=True # ALWAYS extract landmarks for alignment during registration
        )

        # Check if any face was processed and has an embedding
        embedding = None
        if all_results:
            # Find the first result that has an embedding
            for res in all_results:
                if res.embedding:
                    embedding = res.embedding
                    print(f"Using embedding from face @ {res.detection.box} for registration.")
                    break

        if embedding is None:
            # If embedding failed for all detected faces, clean up the saved image
            utils.clean_up_image(image_save_path)
            raise HTTPException(status_code=400, detail="Could not process image or detect face for embedding.")

        # Add to registered embeddings
        new_entry = {"embedding": embedding, "image_filename": unique_filename}
        if label in registered_embeddings:
            # Ensure it's a list (should be handled by load_embeddings validation)
             if not isinstance(registered_embeddings.get(label), list):
                 print(f"Warning: Correcting data structure for label '{label}' during registration.")
                 registered_embeddings[label] = []
             registered_embeddings[label].append(new_entry)
        else:
            registered_embeddings[label] = [new_entry]

        print(f"Registered '{label}'. Total embeddings for label: {len(registered_embeddings[label])}")
        persistence.save_embeddings(registered_embeddings, config.EMBEDDINGS_FILE) # Save after modification
        return {"message": f"Face for '{label}' registered successfully.", "filename": unique_filename}

    except HTTPException as http_exc:
         # Re-raise HTTP exceptions directly
         raise http_exc
    except Exception as e:
        print(f"Error during registration: {e}")
        traceback.print_exc()
        # Clean up image if saving succeeded but processing failed later
        if image_save_path:
            utils.clean_up_image(image_save_path)
        raise HTTPException(status_code=500, detail=f"An internal error occurred during registration: {e}")
    finally:
        # Ensure file is closed even if saving/processing fails
        await file.close()


@app.post("/recognize", response_model=models.RecognitionResponse)
async def recognize_face(
    file: UploadFile = File(...),
    extract_landmarks: bool = Query(False, description="Set to true to extract 68-point landmarks") # Query param
    ):
    """Recognizes a face from an uploaded image against registered faces."""
    if not registered_embeddings:
        raise HTTPException(status_code=400, detail="No faces registered yet.")

    image_np_bgr = None # Initialize
    all_face_results = [] # Initialize

    try:
        image_bytes = await file.read()
        image_np_bgr = utils.decode_image(image_bytes)
        if image_np_bgr is None:
             raise HTTPException(status_code=400, detail="Could not decode query image.")

        # Process image to get results for ALL detected faces
        all_face_results = processing.process_image_full(
            image_np_bgr=image_np_bgr,
            detection_session=detection_session,
            embedding_session=embedding_session,
            landmark_session=landmark_session if extract_landmarks else None, # Pass session only if needed
            extract_68_landmarks=extract_landmarks
        )

        if not all_face_results:
             # No faces detected at all.
             raise HTTPException(status_code=400, detail="No faces detected in the query image.")

        # --- Find Best Match Across All Detected Faces ---
        overall_best_match_label = "unknown"
        overall_highest_similarity = -1.0
        overall_best_matching_entry = None # Info about the matched gallery entry
        best_query_face_result = None # Info about the query face that gave the best match

        # Iterate through each detected face in the query image
        for query_face in all_face_results:
            if query_face.embedding is None:
                print(f"Skipping comparison for face @ {query_face.detection.box} (no embedding generated).")
                continue # Skip faces where embedding failed

            query_embedding_np = np.array(query_face.embedding).reshape(1, -1)
            current_face_best_label = "unknown"
            current_face_highest_similarity = -1.0
            current_face_best_entry = None

            # Compare this query face's embedding against all registered embeddings
            for label, entries_list in registered_embeddings.items():
                if not entries_list: continue

                embeddings_for_label = [entry["embedding"] for entry in entries_list if isinstance(entry, dict) and "embedding" in entry]
                if not embeddings_for_label: continue

                embeddings_np = np.array(embeddings_for_label)
                similarities = cosine_similarity(query_embedding_np, embeddings_np)
                max_similarity_for_label = np.max(similarities)
                best_index_for_label = np.argmax(similarities)

                # Check if this label gives a better match *for the current query face*
                if max_similarity_for_label > current_face_highest_similarity:
                    current_face_highest_similarity = max_similarity_for_label
                    current_face_best_label = label
                    current_face_best_entry = entries_list[best_index_for_label]

            # Now, check if the best match for *this query face* is better than the *overall best match* found so far
            if current_face_highest_similarity > overall_highest_similarity:
                overall_highest_similarity = current_face_highest_similarity
                overall_best_match_label = current_face_best_label
                overall_best_matching_entry = current_face_best_entry
                best_query_face_result = query_face # Store the query face that yielded this best match

        # --- Prepare Response ---
        print(f"Overall best recognition result: Label='{overall_best_match_label}', Similarity={overall_highest_similarity:.4f}, Landmarks requested: {extract_landmarks}")

        # Apply threshold to the overall best match found
        if overall_highest_similarity < config.SIMILARITY_THRESHOLD:
            overall_best_match_label = "unknown"
            overall_best_matching_entry = None
            # Keep overall_highest_similarity as is for info

        # Prepare response data based on the overall best match
        matched_embedding_data = overall_best_matching_entry.get("embedding") if overall_best_matching_entry else None
        matched_filename_data = overall_best_matching_entry.get("image_filename") if overall_best_matching_entry else None

        # Get the query embedding and landmarks from the specific query face that resulted in the best match
        # OR from the highest-scoring detected face if no match was found above threshold
        query_embedding_for_response = None
        query_landmarks_5pt_for_response = None
        query_landmarks_68pt_for_response = None

        if best_query_face_result: # If a match (even below threshold) was found
             query_embedding_for_response = best_query_face_result.embedding
             query_landmarks_5pt_for_response = best_query_face_result.detection.landmarks_5pt
             query_landmarks_68pt_for_response = best_query_face_result.landmarks_68pt if extract_landmarks else None
        elif all_face_results: # If no match found at all, but faces were detected
             # Use the highest scoring detected face (index 0) for landmark display
             highest_scoring_face = all_face_results[0]
             query_embedding_for_response = highest_scoring_face.embedding # Might be None
             query_landmarks_5pt_for_response = highest_scoring_face.detection.landmarks_5pt
             query_landmarks_68pt_for_response = highest_scoring_face.landmarks_68pt if extract_landmarks else None
             print("No match above threshold found. Returning landmarks from highest scoring detected face.")


        return models.RecognitionResponse(
            label=overall_best_match_label,
            similarity=float(overall_highest_similarity),
            query_embedding=query_embedding_for_response,
            matched_embedding=matched_embedding_data,
            matched_image_filename=matched_filename_data,
            query_landmarks_5pt=query_landmarks_5pt_for_response,
            query_landmarks_68pt=query_landmarks_68pt_for_response
        )

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions directly
        raise http_exc
    except Exception as e:
        # Handle other unexpected errors
        print(f"Error during recognition: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal error occurred during recognition: {e}")
    finally:
        # Ensure file is closed in all cases (success, HTTPException, other Exception)
        await file.close()


@app.get("/registered", response_model=models.RegisteredLabelsResponse)
async def get_registered_faces():
    """Returns a list of labels with registered faces."""
    return {"labels": list(registered_embeddings.keys())}


@app.delete("/delete_entry")
async def delete_entry(label: str = Query(...), filename: str = Query(...)):
    """
    Deletes a specific registered face entry (embedding and image)
    identified by its label and image filename.
    """
    global registered_embeddings # Ensure we modify the global dict

    if label not in registered_embeddings:
        raise HTTPException(status_code=404, detail=f"Label '{label}' not found.")

    # Find the index of the entry with the matching filename
    entry_index = -1
    entries_list = registered_embeddings[label]
    for i, entry in enumerate(entries_list):
        if isinstance(entry, dict) and entry.get("image_filename") == filename:
            entry_index = i
            break

    if entry_index == -1:
        raise HTTPException(status_code=404, detail=f"Entry with filename '{filename}' not found for label '{label}'.")

    # --- Deletion Process ---
    image_path = os.path.join(config.GALLERY_DIR, filename)
    deleted_entry_info = f"label='{label}', filename='{filename}'" # For logging

    try:
        # 1. Remove entry from the dictionary
        del entries_list[entry_index]
        print(f"Removed entry {deleted_entry_info} from in-memory data.")

        # 2. If the label list is now empty, remove the label itself
        if not entries_list:
            del registered_embeddings[label]
            print(f"Removed empty label '{label}' from in-memory data.")

        # 3. Save the updated embeddings data
        persistence.save_embeddings(registered_embeddings, config.EMBEDDINGS_FILE)

        # 4. Delete the image file
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
                print(f"Deleted image file: {image_path}")
            except OSError as e:
                # Log error but don't necessarily fail the request if DB entry was removed
                print(f"Warning: Could not delete image file {image_path}: {e}")
                # Optionally, you could raise an exception here if file deletion is critical
        else:
            print(f"Warning: Image file not found for deletion: {image_path}")

        return {"message": f"Successfully deleted entry: {deleted_entry_info}"}

    except Exception as e:
        # Attempt to reload embeddings from file in case of partial failure?
        # Or just report the error. For simplicity, report error.
        print(f"Error during deletion of {deleted_entry_info}: {e}")
        traceback.print_exc()
        # Consider reloading embeddings here to ensure consistency if save failed mid-way
        # registered_embeddings = persistence.load_embeddings(config.EMBEDDINGS_FILE)
        raise HTTPException(status_code=500, detail=f"An internal error occurred during deletion: {e}")


@app.get("/gallery_data", response_model=models.GalleryDataResponse)
async def get_gallery_data():
    """Returns the complete dictionary of registered labels and their associated data."""
    # Ensure embeddings are loaded (should be at startup, but double-check)
    # This check might be redundant if load_embeddings is robust at startup
    # if not registered_embeddings and os.path.exists(config.EMBEDDINGS_FILE):
    #     global registered_embeddings
    #     registered_embeddings = persistence.load_embeddings(config.EMBEDDINGS_FILE)
    return {"data": registered_embeddings}


@app.get("/images/{filename:path}") # Use path converter to handle potential subdirs if needed later
async def get_image(filename: str):
    """Serves an image file from the gallery directory using its filename."""
    base_dir = os.path.abspath(config.GALLERY_DIR)
    requested_path = os.path.abspath(os.path.join(base_dir, filename))

    # Security check: Ensure the resolved path is still within the gallery directory
    if not requested_path.startswith(base_dir) or ".." in filename or filename.startswith("/"):
        print(f"Forbidden access attempt: {filename}")
        raise HTTPException(status_code=403, detail="Forbidden: Access denied.")

    if not os.path.isfile(requested_path):
        print(f"Image not found: {filename} (Resolved: {requested_path})")
        raise HTTPException(status_code=404, detail="Image not found.")

    return FileResponse(requested_path)


# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    print("-" * 30)
    print("Starting Face Recognition API Server")
    print(f"Gallery directory: {os.path.abspath(config.GALLERY_DIR)}")
    print(f"Embeddings file: {os.path.abspath(config.EMBEDDINGS_FILE)}")
    print(f"Detection model: {os.path.abspath(config.DETECTION_MODEL_PATH)}")
    print(f"Embedding model: {os.path.abspath(config.EMBEDDING_MODEL_PATH)}")
    print(f"Landmark model: {os.path.abspath(config.LANDMARK_68_MODEL_PATH)}")
    print(f"Similarity threshold: {config.SIMILARITY_THRESHOLD}")
    print(f"ONNX Providers: {config.PROVIDERS}")
    print(f"Loaded {sum(len(v) for v in registered_embeddings.values())} embeddings for {len(registered_embeddings)} labels.")
    print("-" * 30)

    uvicorn.run(app, host="0.0.0.0", port=8000)