
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
        y: Binary labels (1 for ALF_B or ALF_L, 0 for others)
        class_names: List of class names
        y_labels: Original class labels (for reference)
    """
    images = []
    y_labels = []  # Original class labels
    binary_labels = []  # Binary labels (1 for ALF_B/ALF_L, 0 for others)
    
    # Get list of classes (subdirectories)
    class_names = sorted([d for d in os.listdir(dataset_path) 
                         if os.path.isdir(os.path.join(dataset_path, d))])
    
    # Process each class
    for class_name in class_names:
        class_path = os.path.join(dataset_path, class_name)
        
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, img_file)
                
                # Only process ALF_B and ALF_L classes
                if class_name in ['ALF_B', 'ALF_L']:
                    # Preprocess the image
                    processed_img = preprocess_image(img_path, (IMAGE_WIDTH, IMAGE_HEIGHT))
                    
                    # Add to our lists
                    images.append(processed_img)
                    y_labels.append(class_name)
                    # Binary label: 1 for ALF_L, 0 for ALF_B
                    binary_labels.append(1 if class_name == 'ALF_L' else 0)
    
    # Convert to numpy arrays
    images = np.array(images)
    y = np.array(binary_labels)  # Binary labels for training
    
    # Extract features based on the specified method
    if feature_type.lower() == 'hog':
        X = extract_hog_features(images)
    else:  # Default to pixel features
        X = extract_pixel_features(images)
        
    return X, y, class_names, y_labels  # Return both binary and original labels
    
    return X, labels, ['Not ALF_B', 'ALF_B']  # Binary class names

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os

def train_and_evaluate_models(X, y, class_names, test_size=0.2, random_state=42):
    """
    Train and evaluate multiple machine learning models
    Args:
        X: Feature matrix
        y: Target labels
        class_names: List of class names
        test_size: Proportion of test set (default: 0.2)
        random_state: Random seed for reproducibility
    Returns:
        Dictionary containing evaluation metrics for each model
    """
    # Use stratified split to ensure both classes are represented in training and test sets
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    # Get the training and testing indices
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
    print("\nTraining set class distribution:")
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    for label, count in zip(unique_train, counts_train):
        print(f"- Class {label}: {count} samples")
    
    print("\nTest set class distribution:")
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    for label, count in zip(unique_test, counts_test):
        print(f"- Class {label}: {count} samples")
    
    # Check if we have both classes in training set
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training set must contain samples from both classes. "
                        "Please ensure your dataset has at least one sample from each class.")
    
    print(f"\nSplitting dataset into training ({1-test_size:.0%}) and testing ({test_size:.0%}) sets")
    print(f"Training samples: {len(X_train)} (ALF_B: {sum(y_train)}, Not ALF_B: {len(y_train)-sum(y_train)})")
    print(f"Testing samples: {len(X_test)} (ALF_B: {sum(y_test)}, Not ALF_B: {len(y_test)-sum(y_test)})")
    
    # Initialize models with balanced class weights
    models = {
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        # Initialize SVM with class weights to handle imbalanced data
        'Support Vector Machine': SVC(probability=True, class_weight='balanced', kernel='linear'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced')
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities, handling single-class case
        try:
            proba = model.predict_proba(X_test)
            # If we have probabilities for both classes, take the positive class
            if proba.shape[1] > 1:
                y_prob = proba[:, 1]
            else:
                # If only one class, use its probability directly
                y_prob = proba.flatten()
        except (AttributeError, IndexError):
            # If predict_proba is not available or fails, use the predicted class (0 or 1)
            y_prob = y_pred.astype(float)
        
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
    # 1. Set your dataset path and target classes
    DATASET_PATH = 'dataset'
    TARGET_CLASSES = ['ALF_B', 'ALF_L']  # These are the two classes we're classifying between
    
    # 2. Load and preprocess the dataset
    print("="*50)
    print("Kurdish Alphabet Classification - ML Models")
    print("="*50)
    
    print("\nLoading and preprocessing dataset...")
    
    # Try with HOG features first
    try:
        X, y, class_names, y_labels = load_and_preprocess_dataset(DATASET_PATH, feature_type='hog')
        feature_type = 'hog'
    except Exception as e:
        print(f"Error with HOG features: {e}")
        print("Falling back to pixel features...")
        X, y, class_names, y_labels = load_and_preprocess_dataset(DATASET_PATH, feature_type='pixel')
        feature_type = 'pixel'
    
    # 3. Print dataset information
    print(f"\nDataset loaded successfully!")
    print(f"Feature type: {feature_type.upper()}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    
    # Print class distribution
    print("\nClass distribution (ALF_B vs ALF_L):")
    unique_classes, class_counts = np.unique(y_labels, return_counts=True)
    for cls, count in zip(unique_classes, class_counts):
        print(f"- {cls}: {count} samples")
    
    # Print binary class distribution
    print("\nBinary class distribution (0 = ALF_B, 1 = ALF_L):")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        class_desc = "ALF_L" if label == 1 else "ALF_B"
        print(f"- Class {label} ({class_desc}): {count} samples")
    
    # 4. Train and evaluate models
    print("\n" + "="*50)
    print("Training Machine Learning Models")
    print("="*50)
    
    results = train_and_evaluate_models(X, y, class_names)
    
    # 5. Generate and save comprehensive comparison
    print("\n" + "="*50)
    print("Model Comparison")
    print("="*50)
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Generate comparison data
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'Model': name,
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1-Score': result['f1'],
            'Features': feature_type.upper()
        })
    
    # Convert to DataFrame for better display
    comparison_df = pd.DataFrame(comparison_data)
    
    # Format the display DataFrame (without changing the original data)
    display_df = comparison_df.copy()
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    # Print comparison table
    print("\nPerformance Summary:")
    print(display_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].to_string(index=False))
    
    # Save results to CSV
    results_file = 'results/model_comparison.csv'
    comparison_df.to_csv(results_file, index=False)
    print(f"\nDetailed results saved to '{results_file}'")
    
    # Save all metrics to JSON for reporting
    metrics_report = {
        'feature_type': feature_type,
        'models': {}
    }
    
    for name in results:
        metrics_report['models'][name] = {
            'accuracy': results[name]['accuracy'],
            'precision': results[name]['precision'],
            'recall': results[name]['recall'],
            'f1_score': results[name]['f1']
        }
    
    with open('results/metrics_report.json', 'w') as f:
        json.dump(metrics_report, f, indent=4)
    
    # Generate markdown report
    generate_report(comparison_df, feature_type)
    
    print("\nTraining completed! Check the 'results' folder for detailed performance analysis.")

def generate_report(comparison_df, feature_type):
    """Generate a markdown report with model comparison and analysis"""
    # Sort models by F1-Score to get the best performing model
    sorted_models = comparison_df.sort_values('F1-Score', ascending=False)
    
    # Get the best model details
    best_model_info = sorted_models.iloc[0]
    best_model = best_model_info['Model']
    best_f1 = best_model_info['F1-Score']
    best_acc = best_model_info['Accuracy']
    best_precision = best_model_info['Precision']
    best_recall = best_model_info['Recall']
    
    # Get all models for comparison
    models = sorted_models['Model'].tolist()
    
    # Define reasons for model performance (can be customized based on actual results)
    reasons = {
        'K-Nearest Neighbors': 'Performs well with smaller datasets and can capture local patterns',
        'Support Vector Machine': 'Effective in high-dimensional spaces and works well with clear margin of separation',
        'Random Forest': 'Handles non-linear data well and reduces overfitting through ensemble learning'
    }
    
    # Start building the report
    report = """# Kurdish Alphabet Classification - Model Comparison Report

## 1. Performance Comparison

### Metrics Table

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
"""

    # Add model metrics to the table
    for _, row in sorted_models.iterrows():
        report += f"| {row['Model']} | {row['Accuracy']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1-Score']:.4f} |\n"

    report += f"""
## 2. Best Performing Model

The best performing model is **{best_model}** with an F1-Score of {best_f1:.4f}.

## 3. Analysis of Results

### Performance Analysis

1. **{best_model}** performed the best overall with:
   - Highest F1-Score ({best_f1:.4f}), indicating a good balance between precision and recall
   - {best_acc*100:.2f}% accuracy in classification
   - {best_precision*100:.2f}% precision means that when it predicts a class, it's correct {best_precision*100:.1f}% of the time
   - {best_recall*100:.2f}% recall means it identifies {best_recall*100:.1f}% of all instances correctly

### Feature Impact

- The models used {feature_type.upper()} features for classification
- HOG features are particularly effective for character recognition as they capture the shape and gradient information
- The balanced class weights helped in handling any class imbalance in the dataset

### Algorithm Comparison

"""
    
    # Add algorithm comparison
    for i, (_, row) in enumerate(sorted_models.iterrows(), 1):
        model_name = row['Model']
        report += f"{i}. **{model_name}**: {reasons.get(model_name, 'No specific analysis available')}\n"
    
    report += """
## 4. Confusion Matrices

Confusion matrices for each model are saved as PNG files in the current directory. These show the true vs. predicted labels for both classes.

## 5. Conclusion

The results demonstrate that {0} is the most suitable model for recognizing Kurdish letters in our dataset. The high F1-score indicates that the model achieves a good balance between precision and recall, making it reliable for this classification task.
""".format(best_model)
    
    return report

    # Determine best performing models
    sorted_models = comparison_df.sort_values('F1-Score', ascending=False)
    best = sorted_models.iloc[0]
    second = sorted_models.iloc[1] if len(sorted_models) > 1 else None
    third = sorted_models.iloc[2] if len(sorted_models) > 2 else None

    # Reasons for performance
    reasons = {
        'K-Nearest Neighbors': 'Performs well with clear separation between classes but may struggle with high-dimensional feature spaces',
        'Support Vector Machine': 'Effective in high-dimensional spaces and with clear margin of separation, but may be sensitive to the choice of kernel and parameters',
        'Random Forest': 'Robust to noise and works well with a mixture of feature types, but may overfit with small datasets'
    }

    # Fill in the report template
    report = report.format(
        best_model=best['Model'],
        best_f1=best['F1-Score'],
        best_acc=best['Accuracy'],
        best_precision=best['Precision'],
        best_recall=best['Recall'],
        best_reason=reasons.get(best['Model'], 'Performed best on this dataset'),
        second_model=second['Model'] if second is not None else 'N/A',
        second_reason=reasons.get(second['Model'], '') if second is not None else '',
        third_model=third['Model'] if third is not None else 'N/A',
        third_reason=reasons.get(third['Model'], '') if third is not None else ''
    )

    # Save the report
    with open('results/model_analysis.md', 'w') as f:
        f.write(report)
    
    print("\nReport generated: 'results/model_analysis.md'")
    return report

if __name__ == "__main__":
    main()

