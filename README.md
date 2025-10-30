# There are 2 files


## Accessible-Drop_off

This file contains the actual model under /model.
You can view some of the test segments in /test.
In train/ssh.txt, you can view the parameters I used in training the YOLO model, which should be uploaded as an LFS model.

### Requirements
```bash
pip install -r requirements.txt
```
--------
## Pano-to-Single-View Segmentation Pipeline (convert-pano-to-sing)

This file contains scripts to transform panoramic (equirectangular) images and their associated COCO segmentation annotations into multiple standard (perspective) single-view images (Front, Left, Right), ensuring that the segmentation masks are accurately re-projected onto the new views.

### The Pipeline

The pipeline consists of three main steps:

1.  **Preparation:** Place source panoramic images and the original COCO annotation file into the `data/` directory.
2.  **Transformation (`de_project_and_annotate.py`):** Takes the panoramic image and annotations, performs the geometric transformation, and saves the new data.
3.  **Verification (`verify_masks.py`):** Loads the new data and visually confirms the masks are correctly aligned.

---
### Requirements
```bash
pip install -r requirements.txt
```