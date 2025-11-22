


import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

print("All libraries imported successfully!")

#  Dataset Loading Function
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
BATCH_SIZE=32

train_ds = tf.keras.utils.image_dataset_from_directory(
        'dataset\ALF_B',
        labels='inferred',
        label_mode='categorical', # or 'binary' for two classes, 'int' for integer labels
        image_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation='nearest',
        batch_size=BATCH_SIZE,
        shuffle=True
    )

def load_dataset(dataset_path, img_size=(64, 64)):
    """
    Load images from dataset folders
    Args:
        dataset_path: Path to dataset folder containing class subfolders
        img_size: Target size for resizing images
    Returns:
        images: List of preprocessed images
        labels: List of corresponding labels
    """
    images = []
    labels = []

   
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    print(f"Found {len(classes)} classes: {classes}")

    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        print(f"Loading images from {class_name}...")

        image_count = 0
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(class_path, img_file)
                try:
                   
                    img = cv2.imread(img_path)
                    if img is not None:
                        
                        img = cv2.resize(img, img_size)
                        images.append(img)
                        labels.append(class_name)
                        image_count += 1
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")

        print(f"  Loaded {image_count} images from {class_name}")

    return np.array(images), np.array(labels)


#  Preprocessing Functions


def preprocess_images(images):
    """
    Preprocess images: convert to grayscale and normalize
    Args:
        images: Array of color images
    Returns:
        preprocessed: Array of grayscale normalized images
    """
    preprocessed = []

    for img in images:
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        normalized = gray / 255.0
        preprocessed.append(normalized)

    return np.array(preprocessed)

def extract_hog_features(images):
    """
    Extract HOG (Histogram of Oriented Gradients) features
    Args:
        images: Array of grayscale images
    Returns:
        features: Array of HOG feature vectors
    """
    features = []

    for img in images:
        
        hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=False)
        features.append(hog_features)

    return np.array(features)

def extract_pixel_features(images):
    """
    Extract raw pixel values as features
    Args:
        images: Array of grayscale images
    Returns:
        features: Flattened pixel values
    """
    return images.reshape(len(images), -1)


#  Training and Evaluation Functions


def train_and_evaluate(X_train, X_test, y_train, y_test, model, model_name):
    """
    Train a model and evaluate its performance
    Args:
        X_train, X_test: Training and testing features
        y_train, y_test: Training and testing labels
        model: Machine learning model to train
        model_name: Name of the model for display
    Returns:
        results: Dictionary containing evaluation metrics
    """
    print(f"\nTraining {model_name}...")

  
    model.fit(X_train, y_train)

   
    y_pred = model.predict(X_test)

    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"{model_name} - Accuracy: {accuracy:.4f}")

    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'model': model
    }

    return results

def plot_confusion_matrix(cm, classes, model_name):
    """
    Plot confusion matrix as heatmap
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved for {model_name}")


# Main Workflow


def main():
    """
    Main function to run the complete workflow
    """
    print("="*60)
    print("Kurdish Alphabet Image Classification Project")
    print("="*60)


    DATASET_PATH = 'dataset'  
    IMG_SIZE = (64, 64)
    FEATURE_TYPE = 'hog'  
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"\nERROR: Dataset path '{DATASET_PATH}' not found!")
        print("\nPlease create the dataset structure:")
        print("dataset/")
        print("  ├── Alf_E/  (25 images of Kurdish letter ئ)")
        print("  ├── Alf_B/  (25 images of Kurdish letter ب)")
        print("  ├── Alf_J/  (25 images of Kurdish letter ج)")
        print("  └── Alf_R/  (25 images of Kurdish letter ڕ)")
        return


    print("\n" + "="*60)
    print("STEP 1: Loading Dataset")
    print("="*60)
    images, labels = load_dataset(DATASET_PATH, img_size=IMG_SIZE)
    print(f"\nTotal images loaded: {len(images)}")
    print(f"Image shape: {images[0].shape}")
    print(f"Unique classes: {np.unique(labels)}")

    
    print("\n" + "="*60)
    print("STEP 2: Preprocessing Images")
    print("="*60)
    preprocessed_images = preprocess_images(images)
    print(f"Preprocessed images shape: {preprocessed_images.shape}")

 
    print("\n" + "="*60)
    print("STEP 3: Feature Extraction")
    print("="*60)
    if FEATURE_TYPE == 'hog':
        print("Extracting HOG features...")
        features = extract_hog_features(preprocessed_images)
    else:
        print("Using pixel values as features...")
        features = extract_pixel_features(preprocessed_images)

    print(f"Feature shape: {features.shape}")

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

   
    print("\n" + "="*60)
    print("STEP 4: Splitting Dataset")
    print("="*60)
    X_train, X_test, y_train, y_test = train_test_split(
        features, encoded_labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=encoded_labels
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    
    print("\n" + "="*60)
    print("STEP 5: Training Machine Learning Models")
    print("="*60)

    models = [
        (KNeighborsClassifier(n_neighbors=5), "K-Nearest Neighbors (KNN)"),
        (SVC(kernel='rbf', C=1.0, gamma='scale'), "Support Vector Machine (SVM)"),
        (RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE), "Random Forest")
    ]

    results = []
    for model, name in models:
        result = train_and_evaluate(X_train, X_test, y_train, y_test, model, name)
        results.append(result)


    print("\n" + "="*60)
    print("STEP 6: Model Comparison")
    print("="*60)

    
    comparison_data = []
    for result in results:
        comparison_data.append({
            'Model': result['model_name'],
            'Accuracy': f"{result['accuracy']:.4f}",
            'Precision': f"{result['precision']:.4f}",
            'Recall': f"{result['recall']:.4f}",
            'F1-Score': f"{result['f1_score']:.4f}"
        })

    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))

   
    comparison_df.to_csv('model_comparison.csv', index=False)
    print("\nComparison table saved to 'model_comparison.csv'")


    print("\n" + "="*60)
    print("STEP 7: Generating Confusion Matrices")
    print("="*60)

    class_names = label_encoder.classes_
    for result in results:
        plot_confusion_matrix(result['confusion_matrix'], class_names, result['model_name'])

   
    print("\n" + "="*60)
    print("STEP 8: Best Model Analysis")
    print("="*60)

    best_result = max(results, key=lambda x: x['accuracy'])
    print(f"\nBest Performing Model: {best_result['model_name']}")
    print(f"Accuracy: {best_result['accuracy']:.4f}")
    print(f"Precision: {best_result['precision']:.4f}")
    print(f"Recall: {best_result['recall']:.4f}")
    print(f"F1-Score: {best_result['f1_score']:.4f}")

    print("\n" + "="*60)
    print("Project Completed Successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  - model_comparison.csv")
    print("  - confusion_matrix_K-Nearest_Neighbors_(KNN).png")
    print("  - confusion_matrix_Support_Vector_Machine_(SVM).png")
    print("  - confusion_matrix_Random_Forest.png")


if __name__ == "__main__":
    main()
