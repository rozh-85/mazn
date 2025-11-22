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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
import warnings

warnings.filterwarnings('ignore')

print("All libraries imported successfully!")

#  Dataset Loading Function
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
BATCH_SIZE = 32

import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure
from sklearn.preprocessing import StandardScaler


def preprocess_image(image_path, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT)):
    """
    Preprocess a single image
    Args:
        image_path: Path to the image file
        target_size: Tuple of (width, height) to resize the image to
    Returns:
        processed_image: Preprocessed image
    """
    # 1. Read the image
    image = cv2.imread(image_path)

    # 2. Resize the image
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # 3. Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 4. Normalize pixel values to [0, 1]
    normalized = gray / 255.0

    return normalized


def extract_hog_features(images):
    """
    Extract HOG features from a list of images
    Args:
        images: List of grayscale images
    Returns:
        hog_features: Array of HOG features
    """
    hog_features = []
    for img in images:
        # Extract HOG features
        features = hog(img,
                       orientations=9,
                       pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2),
                       visualize=False)
        hog_features.append(features)

    return np.array(hog_features)


def extract_pixel_features(images):
    """
    Extract raw pixel values as features
    Args:
        images: List of grayscale images
    Returns:
        pixel_features: Flattened array of pixel values
    """
    # Reshape each image to 1D array and stack them
    return np.array([img.flatten() for img in images])


def load_and_preprocess_dataset(dataset_path, feature_type='hog'):
    """
    Load and preprocess all images in the dataset
    Args:
        dataset_path: Path to the dataset directory
        feature_type: Type of features to extract ('hog' or 'pixel')
    Returns:
        X: Extracted features
        y: Binary labels (1 for ALF_B, 0 for others)
        class_names: List of class names
    """
    images = []
    labels = []

    # Get list of classes (subdirectories)
    class_names = sorted([d for d in os.listdir(dataset_path)
                          if os.path.isdir(os.path.join(dataset_path, d))])

    # Process each class
    for class_name in class_names:
        class_path = os.path.join(dataset_path, class_name)

        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, img_file)

                # Preprocess the image
                processed_img = preprocess_image(img_path, (IMAGE_WIDTH, IMAGE_HEIGHT))

                # Add to our lists
                images.append(processed_img)
                # Binary label: 1 for ALF_B, 0 for others
                labels.append(1 if class_name == 'ALF_B' else 0)

    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Extract features based on the specified method
    if feature_type.lower() == 'hog':
        X = extract_hog_features(images)
    else:  # Default to pixel features
        X = extract_pixel_features(images)

    return X, labels, ['Not ALF_B', 'ALF_B']  # Binary class names


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def train_and_evaluate_models(X, y, class_names, test_size=0.2, random_state=42):
    """
    Train and evaluate multiple ML models for binary classification

    Args:
        X: Feature matrix
        y: Binary target labels (0 or 1)
        class_names: List of class names ['Not ALF_B', 'ALF_B']
        test_size: Proportion of test set (default: 0.2)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing trained models and their metrics
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\nSplitting dataset into training ({1 - test_size:.0%}) and testing ({test_size:.0%}) sets")
    print(f"Training samples: {len(X_train)} (ALF_B: {sum(y_train)}, Not ALF_B: {len(y_train) - sum(y_train)})")
    print(f"Testing samples: {len(X_test)} (ALF_B: {sum(y_test)}, Not ALF_B: {len(y_test) - sum(y_test)})")

    # Initialize models with balanced class weights
    models = {
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Support Vector Machine': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced')
    }

    results = {}

    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n{'=' * 50}")
        print(f"Training {name}...")

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # Probability of positive class (ALF_B)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_pred': y_pred,
            'y_prob': y_prob
        }

        # Print results
        print(f"\n{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")

        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{name.lower().replace(" ", "_")}.png', dpi=300)
        plt.close()
        print(f"Confusion matrix saved as 'confusion_matrix_{name.lower().replace(' ', '_')}.png'")

    return results


def main():
    """
    Main function to run the complete workflow
    """
    # 1. Set your dataset path
    DATASET_PATH = 'dataset'

    # 2. Load and preprocess the dataset
    print("=" * 50)
    print("Kurdish Alphabet Classification - ML Models")
    print("=" * 50)

    print("\nLoading and preprocessing dataset...")

    # Try with HOG features first
    try:
        X, y, class_names = load_and_preprocess_dataset(DATASET_PATH, feature_type='hog')
        feature_type = 'hog'
    except Exception as e:
        print(f"Error with HOG features: {e}")
        print("Falling back to pixel features...")
        X, y, class_names = load_and_preprocess_dataset(DATASET_PATH, feature_type='pixel')
        feature_type = 'pixel'

    # 3. Print dataset information
    print(f"\nDataset loaded successfully!")
    print(f"Feature type: {feature_type.upper()}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")

    # 4. Train and evaluate models
    print("\n" + "=" * 50)
    print("Training Machine Learning Models")
    print("=" * 50)

    results = train_and_evaluate_models(X, y, class_names)

    # 5. Print final comparison
    print("\n" + "=" * 50)
    print("Model Comparison")
    print("=" * 50)

    comparison = []
    for name, result in results.items():
        comparison.append({
            'Model': name,
            'Accuracy': f"{result['accuracy']:.4f}",
            'Features': feature_type.upper()
        })

    print("\nPerformance Summary:")
    print(pd.DataFrame(comparison).to_string(index=False))

    print("\nTraining completed! Check the saved confusion matrices for detailed performance.")


if __name__ == "__main__":
    main()
