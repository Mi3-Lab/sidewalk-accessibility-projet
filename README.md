# Sidewalk Accessibility Project

A comprehensive machine learning project for assessing sidewalk accessibility using computer vision and vision-language models. This repository leverages data from Project Sidewalk to predict passability for various mobility aids (e.g., wheelchairs, canes) based on street-level images.

## Features

- **Data Processing**: Aggregates crowdsourced accessibility votes with uncertainty handling.
- **Vision-Language Models**: Uses CLIP for zero-shot and supervised classification of sidewalk images.
- **Mobility Aid-Specific Models**: Trains separate models for Cane, Walker, Mobility Scooter, Manual Wheelchair, and Motorized Wheelchair.
- **Uncertainty-Aware Predictions**: Applies policies to accept only high-confidence predictions.
- **YOLO Segmentation**: Detects sidewalk features like curb ramps and obstacles.
- **Panoramic Image Processing**: Converts 360° images to single-view perspectives with re-projected masks.
- **Web Interface**: Flask app for interactive scoring and visualization.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sidewalk-accessibility-project.git
   cd sidewalk-accessibility-project
   ```

2. Set up Python 3.12 virtual environment:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   This installs the latest versions: PyTorch 2.x, Transformers 5.x, Ultralytics (YOLOv12), etc.

3. Download data: Place Project Sidewalk data (CSVs and images) in `data/`. See notebooks for data sources.

## Usage

### Preprocessing Data
Prepare the dataset by aggregating votes and computing uncertainty metrics.

```bash
python src/preprocess.py --votes_csv data/image_selection.csv --output data/tallies.json
```

### Training Models
Train CLIP-based classifiers for each mobility aid.

```bash
python src/train.py --tallies_json data/tallies.json --images_dir data/sidewalk-images --output_dir models/
```

### Inference
Predict accessibility on a new image.

```bash
python src/infer.py --image path/to/image.jpg --models_dir models/
```

### Training YOLO Models
Train YOLOv12 for sidewalk feature detection.

```bash
python src/train_yolo.py --data data.yaml --model yolo12n.pt --epochs 100
```

### Inferring with YOLO
Detect features in images.

```bash
python src/infer_yolo.py --model best.pt --source image.jpg --save
```

## Project Structure

```
.
├── src/                    # Main source code
│   ├── preprocess.py       # Data preprocessing
│   ├── train.py            # Model training
│   └── infer.py            # Inference
├── data/                   # Datasets and processed data
├── models/                 # Trained model artifacts
├── scripts/                # Utility scripts
├── tests/                  # Unit tests
├── accessible-drop-off-CV/ # YOLO-based CV models
├── accessible-drop-off-model/ # ML models for accessibility
├── convert-pano-to-sing/   # Panoramic image processing
├── requirements.txt        # Python dependencies
└── README.md
```

## Contributing

Contributions are welcome! Please open issues or pull requests for improvements.

## License

This project is licensed under the MIT License. See LICENSE for details.

## Acknowledgments

- Data from [Project Sidewalk](https://projectsidewalk.io/).
- Built with CLIP, YOLO, and other open-source tools.