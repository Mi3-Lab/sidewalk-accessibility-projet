from ultralytics import YOLO

model = YOLO("bestv12.pt")
results = model("extra_images/Crosswalk/320.jpg", conf=0.25)
results[0].show()
print("Boxes:", results[0].boxes)
print("Masks:", results[0].masks)
print(model.names)
