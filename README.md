# Aesthetix AI: Facial Symmetry & Aesthetic Rater 🗿

An AI-powered computer vision system that analyzes facial aesthetics and predicts a rating on a 1.0-5.0 scale. Built with PyTorch, utilizing a fine-tuned ResNet18 architecture and Grad-CAM for visual explainability.

**Try the App:** [Link to your Hugging Face Space]

## Overview

Unlike standard face classifiers that just detect identity, Aesthetix AI is a **regression model** trained to quantify subjective facial attractiveness based on the SCUT-FBP5500 Dataset.

It features a complete inference pipeline:

1. **Face Isolation**: Uses Haar Cascades to detect and tightly crop the face.
2. **Semantic Segmentation**: Uses DeepLabV3 to remove background noise (hair/neck masking) to force the model to evaluate facial geometry only.
3. **Scoring Engine**: A ResNet18 CNN fine-tuned to predict a continuous float score.
4. **Explainability**: Generates Grad-CAM heatmaps to visualize exactly which features (eyes, jawline, symmetry) the model focused on.

## Performance

- **Architecture**: ResNet18 (Pre-trained on ImageNet → Fine-tuned)
- **Loss Function**: MSELoss (Mean Squared Error)
- **Optimizer**: Adam (lr=1e-4)
- **Validation Loss**: 0.0858 (MSE)
  - **Interpretation**: The model's predictions are on average within +/- 0.29 points of the human ground truth.

## The Stack

- **PyTorch**: Core deep learning framework.
- **Torchvision**: Pre-trained models (ResNet18, DeepLabV3).
- **OpenCV**: Face detection and image processing.
- **Streamlit**: Interactive web interface.
- **Grad-CAM**: Visual attention mapping.

## Installation & Usage

1. Clone the repo:

   ```bash
   git clone https://github.com/AKMessi/facial-rating-using-cnn.git
   cd facial-beauty-rating-cnns# facial-rating-using-cnn
