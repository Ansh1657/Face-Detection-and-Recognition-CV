# Setup Guide - Face Recognition Pipeline

## Step-by-Step Installation

### 1. Python Installation

**Check Python version:**
```bash
python --version
```

**Minimum requirement:** Python 3.9+

### 2. Project Setup

**Clone and navigate:**
```bash
git clone https://github.com/Ansh1657/face-recognition-pipeline.git
cd face-recognition-pipeline
```

### 3. Virtual Environment

**Create virtual environment:**
```bash
python -m venv venv
```

**Activate:**
- Windows: `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

### 4. Dependencies

**Install all requirements:**
```bash
pip install -r requirements.txt
```

### 5. Dataset Preparation

**Create directory structure:**
```bash
mkdir -p data/train_images
mkdir -p data/test
mkdir -p results
```

**Add training images:**
- Create subfolder for each person
- Add 5-10 images per person
- Use clear, frontal face images

### 6. First Run

**Train the model:**
```bash
python main.py --mode train --dataset ./data/train_images/
```

**Test detection:**
```bash
python main.py --mode detect --input ./data/test/test.jpg --output ./results/output.jpg
```

## Common Issues

### OpenCV Installation Error
```bash
pip uninstall opencv-python
pip install opencv-python --no-cache-dir
```

### Permission Denied
- Run terminal as administrator (Windows)
- Use `sudo` if needed (Mac/Linux)

### Model Not Found
- Always train before detection
- Check file paths are correct

## System Requirements
```
- **RAM:** Minimum 4GB
- **Storage:** 500MB free space
- **OS:** Windows 10+, macOS 10.14+, Ubuntu 18.04+
```
