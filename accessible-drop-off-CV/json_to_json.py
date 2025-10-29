import json
import os
import requests
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()
api_key = os.environ.get('MAPS_JS_API_KEY')
if not api_key:
    print("Error: The 'MAPS_JS_API_KEY' environment variable is not set. Check your .env file.")
    exit()

def process_raw_json(input_json_path, output_json_path):
    """
    Processes the raw JSON data and converts it into a simplified format
    for the prototype, using a Street View image with a larger marker.
    """
    try:
        with open(input_json_path, 'r') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{input_json_path}' was not found.")
        return

    output_data = []

    for feature in raw_data['features']:
        properties = feature.get('properties', {})
        geometry = feature.get('geometry', {})

        label_type = properties.get('label_type')
        coordinates = geometry.get('coordinates')
        label_cluster_id = properties.get('label_cluster_id')
        median_severity = properties.get('median_severity')
        agree_count = properties.get('agree_count', 0)
        disagree_count = properties.get('disagree_count', 0)
        unsure_count = properties.get('unsure_count', 0)
        
        if not label_type or not coordinates:
            continue

        longitude = coordinates[0]
        latitude = coordinates[1]

        total_votes = agree_count + disagree_count + unsure_count
        confidence = agree_count / total_votes if total_votes > 0 else 0

        # Construct the Street View Static API URL with a larger marker.
        # size: Specifies the marker size. Options are 'tiny', 'mid', 'small'.
        # 'large' is not a standard size, but some third-party libraries might use it.
        image_url = f"https://maps.googleapis.com/maps/api/streetview?size=600x400&location={latitude},{longitude}&heading=180&fov=90&marker=size:mid%7Ccolor:red%7Clabel:L%7C{latitude},{longitude}&key={api_key}"
        
        entry = {
            "id": label_cluster_id,
            "image_url": image_url,
            "labels": [label_type],
            "confidence": confidence,
            "median_severity": median_severity,
            "agree_count": agree_count,
            "disagree_count": disagree_count
        }
        output_data.append(entry)

    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Successfully created '{output_json_path}' with {len(output_data)} entries.")

# Set the path to your downloaded raw JSON file
raw_json_file = "raw_label_clusters.json"
output_json_file = "training_data.json"

process_raw_json(raw_json_file, output_json_file)