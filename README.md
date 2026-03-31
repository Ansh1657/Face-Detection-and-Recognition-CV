# Face-Detection-and-Recognition-CV
A command-line Face Detection and Recognition pipeline. Implements Haar Cascades, PCA, and KNN to detect, extract features, and classify faces without GUI dependencies.
# Face Detection and Recognition Pipeline 👤

A command-line face detection and recognition system implementing computer vision fundamentals using Haar Cascades, PCA, and KNN classification.

## ✨ Features

- **Face Detection**: Haar Cascade-based face detection
- **Feature Extraction**: Principal Component Analysis (PCA) for dimensionality reduction
- **Face Recognition**: K-Nearest Neighbors (KNN) classification
- **CLI Interface**: Pure command-line operation without GUI
- **Model Persistence**: Save and load trained models

## 🎯 Computer Vision Concepts

### Feature Extraction
- Haar Cascades for robust face detection
- HOG (Histogram of Oriented Gradients) features

### Machine Learning
- **PCA (Principal Component Analysis)**: Eigenfaces extraction
- **KNN Classifier**: Face recognition and classification
- Dimensionality reduction for efficient processing

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Terminal/command-line environment

### Setup Steps

1. **Clone the repository**
```bash
   git clone https://github.com/Ansh1657/face-recognition-pipeline.git
   cd face-recognition-pipeline
```

2. **Create virtual environment**
```bash
   python -m venv venv
```

3. **Activate virtual environment**
```bash
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
```

4. **Install dependencies**
```bash
   pip install -r requirements.txt
```

## 📁 Project Structure
```
face-recognition-pipeline/
├── main.py                 # Main application
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── .gitignore             # Git ignore rules
├── LICENSE                # MIT License
├── SETUP_GUIDE.md         # Detailed setup instructions
├── data/                  # Dataset directory (not tracked)
│   ├── train_images/      # Training images
│   │   ├── Person1/
│   │   │   ├── img1.jpg
│   │   │   └── img2.jpg
│   │   └── Person2/
│   │       └── img1.jpg
│   └── test/              # Test images
│       └── input.jpg
├── results/               # Output directory (not tracked)
│   └── output.jpg
└── face_model.pkl         # Trained model (not tracked)
```

## 🎓 Usage

### Training the Model

Train the face recognition model on your dataset:
```bash
python main.py --mode train --dataset ./data/train_images/
```

**Dataset Structure:**
- Create subfolders for each person
- Place multiple images of each person in their folder
- Supported formats: JPG, PNG

**Example:**
```
data/train_images/
├── Alice/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
└── Bob/
    ├── photo1.jpg
    └── photo2.jpg
```

### Detecting and Recognizing Faces

Detect and recognize faces in a new image:
```bash
python main.py --mode detect --input ./data/test/input.jpg --output ./results/output.jpg
```

**Output:**
- Annotated image with bounding boxes
- Person names displayed above detected faces
- Saved to specified output path

### Advanced Options

**Custom model path:**
```bash
python main.py --mode train --dataset ./data/train_images/ --model my_model.pkl
python main.py --mode detect --input test.jpg --output result.jpg --model my_model.pkl
```

## 🔧 Implementation Details

### Face Detection
- **Algorithm**: Haar Cascade Classifier
- **Preprocessing**: Grayscale conversion
- **Detection**: Multi-scale sliding window approach

### Feature Extraction
- **Method**: Principal Component Analysis (PCA)
- **Purpose**: Dimensionality reduction
- **Output**: Eigenfaces (principal components)

### Classification
- **Algorithm**: K-Nearest Neighbors (KNN)
- **Distance Metric**: Euclidean distance
- **Training**: Supervised learning on labeled faces

## 📊 Performance Tips

**For Better Accuracy:**
- Use multiple images per person (5-10 recommended)
- Ensure consistent lighting conditions
- Use frontal face images
- Maintain similar image quality across dataset

**For Faster Processing:**
- Resize training images to consistent dimensions
- Reduce PCA components for larger datasets
- Use grayscale images only

## 🚨 Troubleshooting

**"Model file not found"**
- Run training mode first to create the model
- Check the model path specified

**"No faces detected"**
- Ensure image has clear, frontal faces
- Check image quality and lighting
- Verify image path is correct

**Low recognition accuracy**
- Add more training images per person
- Ensure training images are diverse
- Check for consistent lighting

**ImportError for cv2**
```bash
pip install opencv-python --upgrade
```

**Learning Objectives:**
- Apply feature extraction techniques
- Implement dimensionality reduction
- Build classification pipelines
- Understand face recognition systems

## 🔒 Privacy & Ethics

- Use only authorized images
- Respect privacy and consent
- Follow institutional guidelines
- Do not use for surveillance without permission
     

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- Scikit-learn for machine learning algorithms
- VIT Bhopal University - CSE3010 course materials


**Ansh Chhibber**

---

