from ultralytics import YOLO
import os

MODEL_PATH = "bestv11.pt" #bestv11.pt, train2/weights/bestv12.pt

# This path targets the images inside your 'data/test' folder,
# matching the structure provided in your data.yaml.
TEST_IMAGE_SOURCE = "test/images" 

# The name for the folder where the results will be saved under the 'runs' directory
OUTPUT_RUN_NAME = "v12_test_v2"

# --- Main Prediction Logic ---

# Check if the model path exists (optional but good practice)
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model weights not found at {MODEL_PATH}")
    print("Please ensure you have successfully trained your model and updated the MODEL_PATH variable.")
else:
    # Load the trained segmentation model
    model = YOLO(MODEL_PATH)
    
    print(f"Starting inference on images in: {TEST_IMAGE_SOURCE}")
    
    # Run prediction on the entire directory
    # Using the CLI via the Python API for convenience and standard result saving
    model.predict(
        source=TEST_IMAGE_SOURCE, 
        save=True,               # Save output images
        project = "test",
        name=OUTPUT_RUN_NAME,    # Name of the specific run folder
        conf=0.25,               # Adjust confidence threshold
        
        # === Visualization Settings to Show Masks Only ===
        boxes=True,             # Disable bounding boxes
        show=False,              # Do not display results in a window immediately
        save_txt=False           # Do not save predictions as separate .txt files
    )
    
    # --- Completion Message ---
    final_output_path = os.path.join("runs/predict", OUTPUT_RUN_NAME)
    print("\n--- Prediction Complete ---")
    print(f"Results (images with masks) are saved in: {final_output_path}")
    print("Review the images in this folder to assess model performance.")
