import json
import os
import requests
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()
api_key = os.environ.get('MAPS_JS_API_KEY')
if not api_key:
    print("Error: The 'MAPS_JS_API_KEY' environment variable is not set.")
    exit()

# Define the number of images to download to conserve API credits
MAX_IMAGES = 50
IMAGE_SIZE = "640x640" # Common size for YOLO models

# Define class names and create a mapping from string labels to integer IDs
CLASS_NAMES = ["CurbRamp", "NoSidewalk", "NoCurbRamp", "Crosswalk", "Signal", "Obstacle", "SurfaceProblem"]
CLASS_ID_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}

def create_yolo_dataset(raw_geojson_path):
    """
    Downloads images and creates a YOLO-formatted dataset from raw GeoJSON.
    """
    try:
        with open(raw_geojson_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Raw GeoJSON file '{raw_geojson_path}' not found.")
        return

    # Create the necessary directory structure
    base_dir = "yolo_dataset"
    os.makedirs(os.path.join(base_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "labels", "train"), exist_ok=True)

    # Create the data.yaml file
    data_yaml_content = {
        "path": f"../{base_dir}",
        "train": "images/train",
        "val": "images/train",  # For a small dataset, use the same for train/val
        "names": {i: name for i, name in enumerate(CLASS_NAMES)}
    }
    with open(os.path.join(base_dir, "data.yaml"), "w") as f:
        json.dump(data_yaml_content, f, indent=4)
    
    print("Created data.yaml file.")

    downloaded_count = 0
    for feature in data['features']:
        if downloaded_count >= MAX_IMAGES:
            break

        properties = feature.get('properties', {})
        geometry = feature.get('geometry', {})
        
        label_type = properties.get('label_type')
        coordinates = geometry.get('coordinates')
        label_id = properties.get('label_cluster_id')

        # Skip if essential data is missing or label is not in our classes
        if not label_type or label_type not in CLASS_ID_MAP or not coordinates:
            continue

        longitude, latitude = coordinates
        class_id = CLASS_ID_MAP[label_type]

        # Use the Street View Static API to get a panoramic image.
        # Note: The marker parameter is NOT supported by this API.
        image_url = f"https://maps.googleapis.com/maps/api/streetview?size={IMAGE_SIZE}&location={latitude},{longitude}&heading=180&key={api_key}"
        
        # Download the image
        response = requests.get(image_url)
        if response.status_code == 200:
            image_filename = f"image_{label_id}.jpg"
            image_path = os.path.join(base_dir, "images", "train", image_filename)
            with open(image_path, "wb") as f:
                f.write(response.content)
            
            # Create a corresponding label file with a normalized bounding box
            label_filename = f"image_{label_id}.txt"
            label_path = os.path.join(base_dir, "labels", "train", label_filename)
            
            # Since the data only has a point, we'll create a small, fixed-size bounding box
            # around the center of the image. This is a simple approximation.
            normalized_x = 0.5
            normalized_y = 0.5
            normalized_w = 0.1
            normalized_h = 0.1
            
            with open(label_path, "w") as f:
                f.write(f"{class_id} {normalized_x} {normalized_y} {normalized_w} {normalized_h}\n")
            
            downloaded_count += 1
            print(f"Downloaded and processed image {downloaded_count}/{MAX_IMAGES} for ID {label_id}")
        else:
            print(f"Failed to download image for ID {label_id}. Status code: {response.status_code}")

    print(f"\nDataset creation complete. Downloaded {downloaded_count} images.")

if __name__ == "__main__":
    # Ensure your raw GeoJSON data is in a file named 'raw_data.geojson'
    create_yolo_dataset("raw_label_clusters.json")