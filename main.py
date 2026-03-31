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
                
            detected_faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
            
            for (x, y, w, h) in detected_faces:
                face_roi = img[y:y+h, x:x+w]
                resized_face = cv2.resize(face_roi, (100, 100))
                faces.append(resized_face.flatten())
                labels.append(current_label)
            
        current_label += 1

    if len(faces) == 0:
        print("Error: No faces found in the dataset.")
        return

    print("Extracting features and applying dimensionality reduction (PCA)...")

    faces_array = np.array(faces)
    
    # Using 0.95 to keep 95% of the variance, or fallback to min samples/features if very small dataset
    n_comp = min(50, faces_array.shape[0]) if faces_array.shape[0] < 50 else 0.95 
    pca = PCA(n_components=n_comp)
    pca_transformed = pca.fit_transform(faces_array)
    
    print("Training KNN Classifier...")

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(pca_transformed, labels)

    # Save models to disk
    with open(model_output, 'wb') as f:
        pickle.dump({'pca': pca, 'knn': knn, 'label_map': label_map}, f)
        
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
    if img is None:
        print(f"Error: Could not load image from {input_image_path}")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in detected_faces:
        face_roi = gray[y:y+h, x:x+w]
        resized_face = cv2.resize(face_roi, (100, 100))
        flattened_face = resized_face.flatten().reshape(1, -1)
        
        transformed_face = pca.transform(flattened_face)
        predicted_label = knn.predict(transformed_face)[0]
        
        person_name = label_map.get(predicted_label, "Unknown")
        
        # Draw bounding box and label
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, person_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

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
