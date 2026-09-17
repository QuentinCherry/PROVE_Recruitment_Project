from ultralytics import YOLO

model = YOLO("exp.pt")
model.export(format="coreml", imgsz=1280, half=True, nms=True)