from ultralytics import YOLO

model = YOLO("yolov5l-xs-11.pt")
results = model(r"D:\RL-SAHI\data\raw\images\test\0000006_00611_d_0000002.jpg")
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen