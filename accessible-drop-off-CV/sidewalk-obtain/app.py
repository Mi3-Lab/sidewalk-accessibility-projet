# yolo task=segment mode=train model=yolo11m-seg.pt data=data.yaml epochs=100 imgsz=640 batch=16 lr0=0.001 warmup_epochs=10 optimizer=AdamW

import json
import os
import random
from flask import Flask, render_template, request, jsonify

# Import the scoring logic from your scoring.py file
from scoring import calculate_passability_score

app = Flask(__name__)

# This is where your JSON data is located.
# Construct the absolute path to training_data.json
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'training_data.json')

# Load all spot data into memory when the application starts
try:
    with open(DATA_PATH, 'r') as f:
        all_spots = json.load(f)
except FileNotFoundError:
    print("Error: Data file not found.")
    all_spots = []

@app.route('/')
def index():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/api/spot', methods=['GET'])
def get_random_spot():
    """
    Returns a single random spot's data from memory.
    This simulates a new "frame" from the live stream.
    """
    if not all_spots:
        return jsonify({"error": "No data available."}), 500
    random_spot = random.choice(all_spots)
    return jsonify(random_spot)

@app.route('/api/score', methods=['POST'])
def get_score():
    """
    Calculates the score for a given spot and mobility aid.
    The request should contain the spot_data and mobility_aid.
    """
    data = request.get_json()
    spot_data = data.get('spot_data')
    mobility_aid = data.get('mobility_aid')

    if not spot_data or not mobility_aid:
        return jsonify({"error": "Missing spot data or mobility aid"}), 400

    score = calculate_passability_score(spot_data, mobility_aid)

    return jsonify({
        "score": score
    })

if __name__ == '__main__':
    app.run(debug=True)