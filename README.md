# Detecting Pneumonia from RSNA X-Rays using PyTorch and ResNet

This repository provides a complete, end-to-end pipeline for training a deep learning model (resnet18) to detect pneumonia from chest X-ray images using PyTorch Lightning. The code is designed to work with the RSNA Pneumonia Detection Challenge dataset on Kaggle.

This project tackles two key challenges in medical AI: robustly processing complex DICOM files and managing the severe class imbalance common in medical datasets where disease-positive examples are rare.

## Features of Note
- Dynamic DICOM Preprocessing: Automatically detects the stored bit-depth (e.g., 10-bit, 12-bit) of DICOM files and normalizes them correctly, rather than incorrectly assuming 8-bit (/ 255) data.
- PyTorch Lightning Structure: Uses pl.LightningModule for clean, organized, and scalable training code.
- Imbalanced Dataset Handling: Implements BCEWithLogitsLoss with pos_weight to correctly handle the imbalanced nature of the dataset.
- Optimized for Medical Metrics: The model is trained and checkpointed based on the F1-Score, a much more reliable metric than accuracy for this use case.

## Model Training
- Uses mixed-precision training (precision="16-mixed") for 2x speedup.
- Includes a ReduceLROnPlateau learning rate scheduler.
- Optimized DataLoader with dynamic num_workers and pin_memory.
- Model Interpretability (XAI): Generates Class Activation Maps (CAM) using PyTorch Hooks to visualize why the model is making a prediction. This is done cleanly without redefining the model.
- Baseline Model: Uses a pre-trained ResNet-18 as a strong and fast baseline, which can be easily swapped for other architectures (e.g., DenseNet, EfficientNet).
