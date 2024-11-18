# Product Attribute Classification

This repository contains the training and inference code for classifying product attributes into five categories:  
- **Sarees**  
- **Kurtis**  
- **Men's T-Shirts**  
- **Women's T-Shirts**  
- **Women's Tops and Tunics**  

The project leverages the PVTv2 (Pyramid Vision Transformer v2) model for attribute prediction based on product images.

---

## Prerequisites

### Environment Setup

1. Install the necessary libraries by using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
Install PVTv2-b2:

From GitHub: Clone the repository and install:
bash
Copy code
git clone https://github.com/whai362/PVT
OR From Hugging Face: You can also use the pre-trained model from Hugging Face: OpenGVLab/pvt_v2_b2
Hardware Requirements:

The code is optimized for NVIDIA T4 GPUs and has been tested on Kaggle Notebooks.
If running locally, ensure you have a compatible GPU with the necessary CUDA drivers.
Training
Steps for Training
Prepare Your Dataset:

Create a training.csv file. This CSV should contain:
category: The product category (e.g., Sarees, Kurtis, etc.).
len_attributes_category: The number of attributes each product in this category has.
attribute_labels: The list of attribute labels for each product.
Example structure for training.csv:

Organize the Training Images:

Place all training images in the train_images/ directory.
Configure Training in the Notebook:

Open the training_category.ipynb file.
Set the path to your training.csv file.
Set the path to the train_images/ directory where your images are stored.
Run the Training:

The notebook will train the model for 10 epochs using a learning rate of 1e-4.
The AdamW optimizer is used for model training.
The model will be saved at each epoch, and the best performing model based on F1-micro and F1-macro scores will be selected.
The training process may take some time, especially if running on GPU.

Model Checkpoints:

Once training is complete, the best model (based on harmonic F1 scores) will be saved. You will receive the path to the best-performing model for later use in inference.
Inference
Steps for Inference
Prepare Your Test Dataset:

Create a test.csv file with the same structure as the training CSV. This should contain:
category: The product category (same categories as in the training set).
len_attributes_category: The number of attributes per category.
Example structure for test.csv:

Organize the Test Images:

Place all the test images in the test_images/ directory.
Configure Inference in the Notebook:

Open the inference_category.ipynb file.
Set the path to your test.csv file.
Set the path to the test_images/ directory where your test images are stored.
Load the best model paths (saved during the training step). These models were selected based on the best harmonic F1 scores.
Run the Inference:

The notebook will use the trained model to predict the attribute labels for each product in the test dataset.
The predicted labels will be saved in the output file.
Notes
Local Environment Setup:

If running in a local environment, ensure that all dependencies are installed, including PVTv2-b2. You can install it from GitHub or Hugging Face as described in the prerequisites section.
Ensure that you have CUDA set up if you're using a GPU.
Pre-trained Model:

If you don't want to train the model from scratch, you can use the pre-trained model available in this repository:
GitHub Repo: meesho-data-knights/pvt-v2-b2
Hugging Face: OpenGVLab/pvt_v2_b2
Model Training Time:

The training process may take a while depending on the size of the dataset and the hardware you're using (especially on a GPU).
References
PVTv2 GitHub
PVTv2 on Hugging Face
PVTv2 Model in our GitHub Repository
yaml
Copy code

---

### Key Details in the README:
1. **Clear Sections**: Each section (Prerequisites, Training, Inference, etc.) is separated for clarity.
2. **Instructions**: Detailed steps for setting up the dataset, configuring the paths, and running the notebooks.
3. **Model Reference**: Direct links to the PVTv2 model on GitHub and Hugging Face, as well as your own repository for pre-trained models.

You can copy this into your `README.md` file, and it should be ready to use.
