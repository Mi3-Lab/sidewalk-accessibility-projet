from ultralytics import YOLO

model = YOLO("bestv12.pt")
results = model("extra_images/Obstacle/298.jpg", conf=0.25)
results[0].show()
# print("Boxes:", results[0].boxes)
# print("Masks:", results[0].masks)


# Interesting images:
# extra_images/Surfaceproblem/328181.jpg // Bad sidewalk surface
# extra_images/Crosswalk/320.jpg // Good sidewalks
# extra_images/Obstacle/298.jpg // Obstacle image, note cv model can't detect sidewalk