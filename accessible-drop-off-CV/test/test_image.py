from ultralytics import YOLO
import cv2

# Load your trained model (best.pt from runs)
model = YOLO("train2/weights/bestv12.pt")  # adjust path if needed

# Run prediction on a test image
results = model.predict(
    source="test.jpeg",    # path to your test image
    save=False,            # save output with masks/boxes
    save_txt=False,       # set True if you also want YOLO-format labels
    show=False,             # will open a preview window if GUI available
    conf=0.5
)

# Optionally, display result with OpenCV (first prediction only)
res = results[0]
im_array = res.plot(boxes=True)  # plotted image with masks/boxes
cv2.imshow("Result", im_array)
cv2.waitKey(0)
cv2.destroyAllWindows()
