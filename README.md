# Liver-Cancer-Detection-and-Analysis-using-Unet-2D-and-XGboost
Python pipeline for liver cancer detection from CT scans using 2D segmentation and XGBoost. Includes preprocessing, tumor segmentation, feature extraction, classification into benign/malignant, and visualization of results with feature importance and prediction confidence.
This repository presents a Python-based pipeline for liver tumor detection and classification from CT scans. It combines segmentation techniques with machine learning (XGBoost) to help analyze tumor regions, extract features, and classify them as benign or malignant.

📌 Features

Preprocessing: Convert and normalize 3D NIfTI scans into 2D slices.

Segmentation: Traditional image processing simulates UNet-like segmentation to isolate liver tumors.

Feature Extraction: Shape, intensity, and texture features (GLCM, region properties).

Classification: XGBoost model for binary classification (benign vs malignant).

Visualization:

Tumor masks overlaid on CT slices

Feature importance plots

Prediction confidence display

🧩 Tech Stack

Python 3.x

OpenCV – preprocessing & segmentation

scikit-image – region & texture analysis

XGBoost – classification

Matplotlib – visualization

Nibabel – medical imaging (NIfTI format)

🚀 Workflow

Load CT scan volumes and segmentation masks.

Extract and preprocess 2D slices.

Perform tumor segmentation.

Extract shape, intensity, and texture features.

Train and evaluate the XGBoost model.

Visualize predictions and model insights.

📂 Project Structure
├── liver_cancer_detection.py   # Main script
├── Dataset/                    # Place CT scan volumes and segmentation masks here
│   ├── volumes/
│   └── segmentations/
├── README.md                   # Documentation
└── LICENSE                     # License file

⚙️ Installation

Clone this repository and install dependencies:

git clone https://github.com/your-username/liver-cancer-detection.git
cd liver-cancer-detection
pip install -r requirements.txt

▶️ Usage

Place your dataset inside the Dataset/ folder.

Update the dataset path in the script if needed.

Run the pipeline:

python liver_cancer_detection.py

📊 Example Output

Tumor segmentation overlay on CT slice

Feature importance chart (XGBoost)

Prediction result: Benign / Malignant with confidence score

⚖️ License

© Khurram Izhar 2025. All Rights Reserved.
This code is available for viewing and educational purposes only. Redistribution, modification, or commercial use is prohibited without explicit permission.

🙌 Acknowledgements

Medical imaging handled using Nibabel
.

Machine learning powered by XGBoost
.

Image analysis with scikit-image
.
