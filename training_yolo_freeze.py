import argparse
from ultralytics import YOLO

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--freeze", nargs="+", type=int, default=[10], help="Freeze layers: backbone=10, first3=0 1 2")
    args = parser.parse_args()

    # Load a model
    model = YOLO("yolov5s.pt")  # build from YAML and transfer weights

    freeze = [f"model.{x}." for x in (args.freeze if len(args.freeze) > 1 else range(args.freeze[0]))]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        if any(x in k for x in freeze):
            print(f'Freezing {k}')
            v.requires_grad = False

    # Train the model
    results = model.train(data="coco128.yaml", epochs=10, imgsz=2048, batch=1, device='xpu')
