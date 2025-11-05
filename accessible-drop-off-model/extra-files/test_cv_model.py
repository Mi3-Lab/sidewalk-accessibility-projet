from ultralytics import YOLO

model = YOLO("bestv12.pt")
results = model("Project Sidewalk Data/sidewalk-images/gsv-cdmx-44798-1-4.png", conf=0.25)
results[0].show()
print("Boxes:", results[0].boxes)
print("Masks:", results[0].masks)
print(model.names)
