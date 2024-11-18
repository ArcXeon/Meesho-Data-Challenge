# Predict Attribute from Product Images (Team Name : Data_Knights)

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
Dataset
You can download the dataset for this project from the Kaggle competition:
Visual Taxonomy - Kaggle Competition

# Visual Taxonomy - Product Attribute Prediction

This repository contains code and instructions for the **Visual Taxonomy** competition, where the goal is to predict product attributes based on images. The dataset includes images for different product categories such as Sarees, Kurtis, Men’s T-Shirts, Women’s T-Shirts, and Women’s Tops and Tunics. Each product has associated attributes that need to be predicted based on the images.

## Dataset Overview

The dataset includes the following files and directories:

- **train.csv**: Contains metadata for the training images, including the product category, number of attributes, and their labels.
- **test.csv**: Contains metadata for the test images, including the product category and the number of attributes.
- **train_images/**: Directory with the training images.
- **test_images/**: Directory with the test images.

## Training

### Steps for Training

1. **Prepare Your Dataset**:

   Create a `training.csv` file with the following structure:
   
   | category          | len                     | attr_i  |
   |-------------------|-------------------------|-------------------|
   | Sarees            | 10                       | ['Label1', 'Label2', ...] |
   | Kurtis            | 9                       | ['Label1', 'Label2', ...] |
   | Men T-Shirts      | 5                       | ['Label1', 'Label2', ...] |
   | ...               | ...                     | ...               |

2. **Organize the Training Images**:
   
   Place all the training images in the `train_images/` directory.

3. **Configure Training in the Notebook**:

   Open the `training_category.ipynb` notebook and set the following paths:
   - `training.csv`: Path to your training CSV file.
   - `train_images/`: Path to the directory containing the training images.
   - After this , training code will do training of that category, similary we can do train for every category .

4. **Run the Training**:

   - The notebook will train the model for 10 epochs with a learning rate of `1e-4`.
   - The AdamW optimizer will be used.
   - The model will be saved at each epoch, and the best-performing model based on **F1-micro** and **F1-macro** scores will be selected.
   

5. **Model Checkpoints**:

   - After training is complete, the best model (based on harmonic F1 scores) will be saved.
   - The path to the best-performing model will be provided for later use in inference.
   - So we have now all the model paths , which are required for inference for test data.

---

## Inference

### Steps for Inference

1. **Prepare Your Test Dataset**:

   Create a `test.csv` file with the same structure as the `training.csv` file, which should contain:
   
   | category          | len_attributes_category | attr_i  |
   |-------------------|-------------------------|-------------------|
   | Sarees            | 10                       | ['Label1', 'Label2', ...] |
   | Kurtis            | 9                       | ['Label1', 'Label2', ...] |
   | Men T-Shirts      | 5                       | ['Label1', 'Label2', ...] |
   | ...               | ...                     | ...               |

2. **Organize the Test Images**:
   
   Place all the test images in the `test_images/` directory.

3. **Configure Inference in the Notebook**:

   Open the `inference_category.ipynb` notebook and set the following paths:
   - `test.csv`: Path to your test CSV file.
   - `test_images/`: Path to the directory containing the test images.
   -  After this , inference code will do prediction of attributes labels of that category, similary we can do for every category 

   Also, load the best model paths (saved during the training step). These models were selected based on the best harmonic F1 scores.

4. **Run the Inference**:

   The notebook will use the trained model to predict the attribute labels for each product in the test dataset. 

5. **Output**:

   The predicted labels of attributes of each category is saved in the output file from each category inference code. Then we use just merge the output of every category and get the desired submission file       
   (combine-submissions.ipynb)

---

## Notes

- The training process will take some time (27-28 hours for whole training dataset for all categories). 
- Ensure that your training and test image directories are properly organized and the paths are correctly set in the notebooks.
- The best model will be selected based on the F1 score, and its path will be provided for inference.



### Key Details in the README:
1. **Clear Sections**: Each section (Prerequisites, Training, Inference, etc.) is separated for clarity.
2. **Instructions**: Detailed steps for setting up the dataset, configuring the paths, and running the notebooks.
3. **Model Reference**: Direct links to the PVTv2 model on GitHub and Hugging Face, as well as in our repository for pre-trained models.

## Notes

### Local Environment Setup

- If running in a local environment, ensure that all dependencies are installed, including **PVTv2-b2**. You can install it from GitHub or Hugging Face as described in the **Prerequisites** section.
- Ensure that you have **CUDA** set up if you're using a GPU.

### Pre-trained Model

- If you don't want to train the model from scratch, you can use the pre-trained model available in this repository:
  - **GitHub Repo**: [meesho-data-knights/pvt-v2-b2](https://github.com/meesho-data-knights/pvt-v2-b2)
  - **Hugging Face**: [OpenGVLab/pvt_v2_b2](https://huggingface.co/OpenGVLab/pvt_v2_b2)

### Model Training Time

- The training process may take a while depending on the size of the dataset and the hardware you're using (especially on a GPU).

---

## References

- [PVTv2 GitHub](https://github.com/meesho-data-knights/pvt-v2-b2)
- [PVTv2 on Hugging Face](https://huggingface.co/OpenGVLab/pvt_v2_b2)
- [PVTv2 Model in our GitHub Repository](https://github.com/meesho-data-knights/pvt-v2-b2)

