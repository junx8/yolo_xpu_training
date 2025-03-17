from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolov8s.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data="coco128.yaml", epochs=100, imgsz=2048, batch=8, device='xpu')
