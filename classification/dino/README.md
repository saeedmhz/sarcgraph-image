# DINO Feature Extraction and Classification

This folder contains scripts and models related to feature extraction using the DINOv2 model and classification with a trained MLP.

## Files:
- **`trained_models/`**: Contains trained MLP models (`model_1.pth` to `model_5.pth`), each corresponding to a unique train-test split.
- **`dino_feature_vectors.npy`**: The extracted feature vectors from the DINOv2 model for downstream classification tasks.
- **`feature-generator.py`**: Code for generating feature vectors using DINOv2.
- **`features-pca.png`**: PCA plot of the extracted features, visualizing clusters based on labels.
- **`features-pca-plot.py`**: Code for generating the PCA visualization of the feature vectors.
- **`model-training.py`**: Code for training and evaluating MLP classifiers on DINO feature vectors.
