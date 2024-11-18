Product Attribute Classification
This repository contains the training and inference code for classifying product attributes into five categories:

Sarees
Kurtis
Men's T-Shirts
Women's T-Shirts
Women's Tops and Tunics
The project leverages the PVTv2 (Pyramid Vision Transformer v2) model for attribute prediction based on product images.

Prerequisites
Environment Setup
Ensure all required libraries are installed. Install dependencies from requirements.txt:

bash
Copy code
pip install -r requirements.txt
Additionally, install PVTv2-b2 from:

GitHub: https://github.com/whai362/PVT
OR
Hugging Face: OpenGVLab/pvt_v2_b2
Hardware Requirements
The code is optimized for NVIDIA T4 GPUs and is tested on Kaggle Notebooks.

Training
Steps
Prepare your dataset:

training.csv: Should contain the following columns:
category: The product category.
len_attributes_category: Number of attributes per category.
attribute_labels: Labels for the attributes.
Organize the images:

train_images/: Directory containing training images.
Open the training_category.ipynb file and set the following:

Path to training.csv.
Path to train_images/.
Run the notebook. Training will:

Use 10 epochs with a learning rate of 1e-4.
Use the AdamW optimizer.
Save the model checkpoints for the best harmonic mean of F1-micro and F1-macro scores.
Inference
Steps
Prepare your dataset:

test.csv: Should contain the following columns:
category: The product category.
len_attributes_category: Number of attributes per category.
Organize the images:

test_images/: Directory containing test images.
Open the inference_category.ipynb file and set the following:

Path to test.csv.
Path to test_images/.
Load the model paths (saved during training) with the best harmonic F1 scores.
Run the notebook to predict the attribute classes for the test dataset.

Notes
If using a local environment, ensure all dependencies are installed, including PVTv2-b2. You can:

Clone the GitHub repository: https://github.com/whai362/PVT
OR directly use the Hugging Face model: OpenGVLab/pvt_v2_b2.
The pre-trained PVTv2-b2 model is also available in this repository under:

GitHub Repo: meesho-data-knights/pvt-v2-b2
References
PVTv2 GitHub
PVTv2 on Hugging Face
This README provides all the steps for training and inference. Feel free to reach out if you have further questions or encounter any issues.
