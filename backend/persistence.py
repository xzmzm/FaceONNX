import os
import json
from typing import Dict, List

# Structure: { "label": [ {"embedding": [float_list], "image_filename": "unique_image.jpg"}, ... ] }
EmbeddingsData = Dict[str, List[Dict[str, list | str]]]

def load_embeddings(embeddings_file: str) -> EmbeddingsData:
    """Loads registered embeddings from the JSON file."""
    registered_embeddings: EmbeddingsData = {}
    if os.path.exists(embeddings_file):
        try:
            with open(embeddings_file, 'r') as f:
                loaded_data = json.load(f)
                # Basic validation
                if isinstance(loaded_data, dict):
                    valid_structure = True
                    total_embeddings = 0
                    for label, entries in loaded_data.items():
                        if isinstance(entries, list):
                            total_embeddings += len(entries)
                            for entry in entries: # Check for required keys
                                if not (isinstance(entry, dict) and "embedding" in entry and "image_filename" in entry):
                                    valid_structure = False
                                    print(f"Warning: Invalid entry structure found for label '{label}' in {embeddings_file}: {entry}")
                                    break
                        else:
                            valid_structure = False
                            print(f"Warning: Invalid data type for label '{label}' in {embeddings_file}. Expected list, got {type(entries)}.")
                        if not valid_structure:
                            break

                    if valid_structure:
                        registered_embeddings = loaded_data
                        print(f"Loaded {total_embeddings} embeddings/images for {len(registered_embeddings)} labels from {embeddings_file}")
                    else:
                        print(f"Warning: Invalid data structure found in {embeddings_file}. Starting with empty embeddings.")
                        registered_embeddings = {}
                else:
                    print(f"Warning: Invalid format in {embeddings_file} (expected a dictionary). Starting with empty embeddings.")
                    registered_embeddings = {}

        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading embeddings from {embeddings_file}: {e}. Starting with empty embeddings.")
            registered_embeddings = {}
    else:
        print(f"Embeddings file ({embeddings_file}) not found. Starting with empty embeddings.")
        registered_embeddings = {}
    return registered_embeddings

def save_embeddings(embeddings_data: EmbeddingsData, embeddings_file: str):
    """Saves the current registered embeddings to the JSON file."""
    try:
        # Ensure parent directory for embeddings file exists
        parent_dir = os.path.dirname(embeddings_file)
        if parent_dir and not os.path.exists(parent_dir):
             os.makedirs(parent_dir, exist_ok=True)
             print(f"Created directory for embeddings file: {parent_dir}")

        with open(embeddings_file, 'w') as f:
            json.dump(embeddings_data, f, indent=4)
        print(f"Saved {sum(len(v) for v in embeddings_data.values())} embeddings for {len(embeddings_data)} labels to {embeddings_file}")
    except IOError as e:
        print(f"Error saving embeddings to {embeddings_file}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving embeddings: {e}")