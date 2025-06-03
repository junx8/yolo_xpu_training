# YOLO training with XPU

## Intel xpu training patch for ultralytics

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/test/xpu

git clone https://github.com/junx8/yolo_xpu_training.git
git submodule update --init --recursive

cd ultralytics
git am ../0001-enable-intel-xpu.patch
pip install -e .
```

## Training

```
cp ../training_yolo.py .  # workaround for issue "ImportError: cannot import name 'YOLO' from 'ultralytics' (unknown location)"
python training_yolo.py
```

### Training with frozen layers

```
python training_yolo_freeze.py

or

python training_yolo_freeze.py -f N

or

python training_yolo_freeze.py -f 1 2 3

```
