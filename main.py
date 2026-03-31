import argparse
import cv2
import os
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

def train_model(dataset_path, model_output):
    print(f"Starting training process using dataset at: {dataset_path}")
    
    faces = []
    labels = []
    label_map = {}
    current_label = 0
    
    # Initialize Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue
            
        label_map[current_label] = person_name
        
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
                
            # STUDENT IMPLEMENTATION REQUIRED:
            # 1. Use face_cascade.detectMultiScale to find faces
            # 2. Crop the face region
            # 3. Resize to a fixed dimension (e.g., 100x100)
            # 4. Flatten the array and append to 'faces'
            # 5. Append 'current_label' to 'labels'
            pass
            
        current_label += 1

    print("Extracting features and applying dimensionality reduction (PCA)...")
    # STUDENT IMPLEMENTATION REQUIRED:
    # 1. Convert 'faces' to a numpy array
    # 2. Initialize PCA and fit_transform the data
    
    print("Training KNN Classifier...")
    # STUDENT IMPLEMENTATION REQUIRED:
    # 1. Initialize KNeighborsClassifier
    # 2. Fit the classifier using the PCA-transformed data and 'labels'

    # Save models to disk
    with open(model_output, 'wb') as f:
        # pickle.dump({'pca': pca, 'knn': knn, 'label_map': label_map}, f)
        pass
        
    print(f"Training complete. Model saved to {model_output}")

def detect_faces(input_image_path, output_image_path, model_path):
    print(f"Loading model from {model_path}...")
    
    if not os.path.exists(model_path):
        print("Error: Model file not found. Run training first.")
        return

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        
    pca = model_data['pca']
    knn = model_data['knn']
    label_map = model_data['label_map']

    img = cv2.imread(input_image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # STUDENT IMPLEMENTATION REQUIRED:
    # 1. Detect faces in the 'gray' image
    # 2. Loop through bounding boxes (x, y, w, h)
    # 3. Crop, resize, and flatten the detected face region
    # 4. Use pca.transform() on the flattened region
    # 5. Use knn.predict() to get the label
    # 6. Retrieve the person's name from label_map
    # 7. Draw a rectangle and putText on 'img'

    cv2.imwrite(output_image_path, img)
    print(f"Detection complete. Result saved to {output_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI Face Detection and Recognition Pipeline")
    parser.add_argument("--mode", choices=["train", "detect"], required=True, help="Run mode: 'train' or 'detect'")
    parser.add_argument("--dataset", type=str, help="Path to training dataset directory")
    parser.add_argument("--input", type=str, help="Path to input image")
    parser.add_argument("--output", type=str, help="Path to save output image")
    parser.add_argument("--model", type=str, default="face_model.pkl", help="Path to model file")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        if not args.dataset:
            print("Error: --dataset required for training.")
        else:
            train_model(args.dataset, args.model)
            
    elif args.mode == "detect":
        if not args.input or not args.output:
            print("Error: --input and --output required for detection.")
        else:
            detect_faces(args.input, args.output, args.model)
