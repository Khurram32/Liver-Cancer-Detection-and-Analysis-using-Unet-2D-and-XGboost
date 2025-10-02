# Description: A Python script to detect and classify liver tumors from CT scans
#              using traditional Computer Vision for segmentation and an XGBoost model for classification.
#              (This version does NOT require TensorFlow.)

# --- 1. Import Necessary Libraries ---
print("Importing libraries...")
import os
import numpy as np
import pandas as pd
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from skimage.feature import graycomatrix, graycoprops
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
print("Libraries imported successfully.")

# --- 2. Configuration and Constants ---
IMG_HEIGHT = 256
IMG_WIDTH = 256

# --- 3. Data Loading and Preprocessing ---
def preprocess_data(image_dir, mask_dir, target_size=(IMG_WIDTH, IMG_HEIGHT)):
    """
    Loads 3D NIfTI files, extracts 2D slices, resizes, and normalizes them.
    """
    print("Starting data preprocessing...")
    images = []
    masks = []

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.nii', '.nii.gz'))])
    print(f"Found {len(image_files)} files to process.")

    for filename in image_files:
        try:
            img_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, filename.replace('volume', 'segmentation'))

            if not os.path.exists(mask_path):
                print(f"Warning: Mask not found for {filename}. Skipping.")
                continue

            img_nii = nib.load(img_path)
            mask_nii = nib.load(mask_path)
            
            img_data = img_nii.get_fdata()
            mask_data = mask_nii.get_fdata()
            
            for i in range(img_data.shape[2]):
                slice_img = img_data[:, :, i]
                slice_mask = mask_data[:, :, i]

                if np.sum(slice_mask) == 0:
                    continue

                slice_img_resized = cv2.resize(slice_img, target_size, interpolation=cv2.INTER_AREA)
                slice_mask_resized = cv2.resize(slice_mask, target_size, interpolation=cv2.INTER_NEAREST)
                
                min_val, max_val = np.min(slice_img_resized), np.max(slice_img_resized)
                if max_val - min_val > 1e-6:
                    slice_img_normalized = (slice_img_resized - min_val) / (max_val - min_val)
                else:
                    slice_img_normalized = slice_img_resized

                images.append(slice_img_normalized)
                masks.append((slice_mask_resized > 0).astype(np.uint8))
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    print(f"Preprocessing complete. Loaded {len(images)} slices.")
    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.uint8)

# --- 4. Traditional CV Segmentation ---
def segment_tumor_traditional(image_slice):
    img_8bit = (image_slice * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(img_8bit, (5, 5), 0)
    _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(opening, dtype=np.uint8)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(final_mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
    return (final_mask / 255.0).astype(np.uint8)

# --- 5. Feature Extraction (CRITICAL UPDATE HERE) ---
def extract_features(image_slice, predicted_mask):
    """
    Extracts shape, intensity, and texture features from the segmented tumor region.
    """
    binary_mask = (predicted_mask > 0.5).astype(np.uint8)
    if np.sum(binary_mask) == 0: return None
    
    props = regionprops(binary_mask, intensity_image=image_slice)
    if not props: return None
    
    tumor_props = props[0]
    
    # 1. Shape-based Features
    area = tumor_props.area
    perimeter = tumor_props.perimeter
    eccentricity = tumor_props.eccentricity
    circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
    
    # 2. Intensity-based Features
    mean_intensity = tumor_props.mean_intensity
    std_intensity = np.std(image_slice[binary_mask == 1])
    
    # --- CORRECTED TEXTURE FEATURE LOGIC ---
    # Create a bounding box around the tumor to define a Region of Interest (ROI)
    minr, minc, maxr, maxc = tumor_props.bbox
    roi = image_slice[minr:maxr, minc:maxc]
    
    # Calculate GLCM only on the 8-bit ROI
    glcm_img = (roi * 255).astype(np.uint8)
    if glcm_img.size == 0: return None # Skip if ROI is empty

    glcm = graycomatrix(glcm_img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    return {
        'area': area, 'perimeter': perimeter, 'eccentricity': eccentricity,
        'circularity': circularity, 'mean_intensity': mean_intensity,
        'std_intensity': std_intensity, 'contrast': contrast,
        'correlation': correlation, 'energy': energy, 'homogeneity': homogeneity
    }

# --- 6. Visualization Functions ---
def plot_feature_importance(model, feature_names):
    """ Creates a bar plot of feature importances from the trained XGBoost model. """
    importances = model.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(10, 8))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='skyblue', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.show()

def visualize_prediction(image, mask, class_label, probability):
    """ Displays the original image, its segmentation mask, and the prediction. """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original CT Slice')
    axes[0].axis('off')

    axes[1].imshow(image, cmap='gray')
    axes[1].imshow(mask, cmap='jet', alpha=0.5) # Overlay mask
    axes[1].set_title('Predicted Tumor Mask')
    axes[1].axis('off')

    fig.suptitle(f'Prediction: {class_label} (Confidence: {probability*100:.2f}%)', fontsize=16, weight='bold')
    plt.show()

# --- 7. Main Execution Workflow ---
if __name__ == "__main__":
    BASE_DATASET_PATH = r'C:\Users\Cyphe\Downloads\Dataset'
    IMAGE_DIR = os.path.join(BASE_DATASET_PATH, 'volumes')
    MASK_DIR = os.path.join(BASE_DATASET_PATH, 'segmentations')

    if not os.path.exists(IMAGE_DIR) or not os.path.exists(MASK_DIR):
        print("!!! ERROR: Data directory not found.")
    else:
        # === Part 1: Data Loading & Feature Extraction ===
        print("\n--- Part 1: Loading Data and Extracting Features ---")
        images, _ = preprocess_data(IMAGE_DIR, MASK_DIR)
        
        features_list = []
        if images.size > 0:
            for img_slice in images:
                predicted_mask = segment_tumor_traditional(img_slice)
                if np.sum(predicted_mask) > 0:
                    features = extract_features(img_slice, predicted_mask)
                    if features:
                        features_list.append(features)
        print(f"Extracted features for {len(features_list)} detected regions.")

        # === Part 2: Classification with XGBoost ===
        if not features_list:
            print("Could not extract any features. The segmentation method may need tuning.")
        else:
            print("\n--- Part 2: Training XGBoost for Classification ---")
            feature_names = list(features_list[0].keys())
            X_features = np.array([[f[name] for name in feature_names] for f in features_list])

            print("\n!!! SIMULATING CLASSIFICATION LABELS FOR DEMONSTRATION !!!")
            np.random.seed(42)
            y_labels_simulated = np.random.randint(0, 2, size=len(features_list))

            # Updated classifier call for modern XGBoost versions
            xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
            xgb_model.fit(X_features, y_labels_simulated)
            
            preds = xgb_model.predict(X_features)
            print(f"XGBoost training accuracy (on simulated data): {accuracy_score(y_labels_simulated, preds) * 100:.2f}%")
            
            print("\nDisplaying feature importance plot...")
            plot_feature_importance(xgb_model, feature_names)
            
            # === Part 3: Full Prediction Pipeline Example ===
            print("\n--- Part 3: Running Full Prediction Pipeline on a Sample ---")
            if len(images) > 10:
                _, X_val_imgs = train_test_split(images, test_size=0.2, random_state=42)
                sample_image = X_val_imgs[10]

                predicted_mask = segment_tumor_traditional(sample_image)
                sample_features_dict = extract_features(sample_image, predicted_mask)

                if sample_features_dict:
                    sample_features_vec = np.array([list(sample_features_dict.values())])
                    prediction = xgb_model.predict(sample_features_vec)
                    prediction_proba = xgb_model.predict_proba(sample_features_vec)
                    
                    class_label = "Malignant" if prediction[0] == 1 else "Benign"
                    confidence = np.max(prediction_proba)
                    
                    print(f"\nPrediction for sample image:")
                    print(f" -> Predicted Class: ** {class_label} ** (Confidence: {confidence*100:.2f}%)")
                    
                    print("\nDisplaying sample prediction plot...")
                    visualize_prediction(sample_image, predicted_mask, class_label, confidence)
                else:
                    print("\nNo tumor was detected in the sample image.")
            else:
                print("\nNot enough images to create a validation sample for visualization.")