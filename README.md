---

# Install YOLOv5 on JetPack 4.6.1 (Jetson Nano)

This guide helps you install a compatible YOLOv5 environment on **Jetson Nano (JetPack 4.6.1 / CUDA 10.2 / Python 3.6)**.

---

## 1. Update System

```bash
sudo apt update
sudo apt upgrade
```

---

## 2. Install Required System Packages

```bash
sudo apt install python3-pip git libopenblas-base libopenmpi-dev
sudo apt install libjpeg-dev zlib1g-dev
```

---

## 3. Upgrade pip

```bash
python3 -m pip install --upgrade pip
```

---

## 4. Install PyTorch (JetPack 4.6.1 Compatible)

```bash
sudo apt install python3-pip
pip3 install numpy==1.19.4
```

---

## 5. Install Correct Jetson PyTorch

For **JetPack 4.6.1 (CUDA 10.2)**:

### Install dependencies

```bash
sudo apt install libopenblas-base libopenmpi-dev -y
```

### Install NVIDIA PyTorch wheel

```bash
wget https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch/torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
```

---

## 6. Install torchvision (Compatible Version)

For **torch 1.10.0**:

```bash
sudo apt-get install libjpeg-dev zlib1g-dev

git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
cd torchvision

export BUILD_VERSION=0.11.1
python3 setup.py install --user
```

---

## 7. Fix OpenCV (if `import cv2` fails)

 **Important:** Do NOT install OpenCV via pip

```bash
sudo apt update
sudo apt install python3-opencv -y
```

### Test installation:

```bash
python3 -c "import cv2; print(cv2.__version__)"
```

---

## 8. Clone Compatible YOLOv5 Version

 Latest YOLOv5 does **NOT** support Python 3.6

```bash
git clone https://github.com/kritsana3376/Fall_Detection_YOLOv5.git
cd Fall_Detection_YOLOv5
```

---

##  9. Install Python Dependencies

```bash
sudo pip3 install --upgrade pip
sudo python3 -m pip install -r requirements.txt
```

### If errors occur, install manually:

```bash
pip3 install matplotlib==3.3.4
pip3 install opencv-python
pip3 install PyYAML
pip3 install tqdm
pip3 install seaborn
```

---

##  10. Enable Max Performance (IMPORTANT)

Run **before detection**:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

---

##  11. Run YOLOv5

```bash
python3 main.py
```

---

#  Install ONNX & Export TensorRT Engine (Jetson Nano - JetPack 4.6.1)

This guide helps you install **ONNX correctly on Jetson Nano (ARM, Python 3.6)** and export YOLOv5 to **TensorRT (.engine)**.

---

##  1. Clone Compatible YOLOv5 Version

 Latest YOLOv5 does **NOT support Python 3.6**

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
git reset --hard 9bcc32a
```

---

##  2. Install Dependencies

```bash
sudo apt update
sudo apt install -y python3-pip libprotobuf-dev protobuf-compiler
```

---

##  3. Upgrade pip (IMPORTANT)

```bash
pip3 install --upgrade pip
```

---

##  4. Install ONNX

```bash
pip3 install onnx
```
---
### If installation fails

```bash
pip3 install onnx==1.10.0
```
---
## Option: Build ONNX from Source

### Step 1: Install build tools

```bash
sudo apt install -y git cmake build-essential
```

---

### Step 2: Clone ONNX

```bash
git clone https://github.com/onnx/onnx.git
cd onnx
git checkout v1.10.0
```

---

### Step 3: Build

 Limit jobs to avoid crash on Nano

```bash
export MAX_JOBS=1
python3 setup.py install
```

---

## IMPORTANT: Prevent Out-of-Memory (Nano 2GB)

Before building ONNX:

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## Test Installation

```bash
python3 -c "import onnx; print(onnx.__version__)"
```

---

## Export YOLOv5 to TensorRT (.engine)

```bash
python export.py --weights yolov5s.pt --img 640 --device 0 --include engine
```
---

## Why Version Matters for ONNX

* New ONNX versions often **do NOT support ARM well**
* Jetson Nano works best with:
 **ONNX 1.10 – 1.12**
---

## Notes

* Designed for **Jetson Nano (JetPack 4.6.1, CUDA 10.2)**
* Uses **official NVIDIA PyTorch build**
* Python **must be 3.6**
* YOLOv5 version pinned for compatibility
* Avoid upgrading Python or PyTorch (will break setup)
* ONNX version is critical for ARM compatibility
* Use swapfile when building to avoid memory crash

---
## Collaborators

This project was developed in collaboration with:

- Kritsana Netpugdee  
- Purin Chirapornchai  
- Jakapat Dungdee
