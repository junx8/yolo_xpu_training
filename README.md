# yolo_xpu_training

## intel xpu training patch for ultralytics

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/test/xpu

git clone https://github.com/junx8/yolo_xpu_training.git
git submodule update --init --recursive

cd ultralytics
git am ../0001-enable-intel-xpu.patch
pip install -e .
```

## training

```
cp ../training_yolo.py .  # workaround for issue "ImportError: cannot import name 'YOLO' from 'ultralytics' (unknown location)"
python training_yolo.py
```
