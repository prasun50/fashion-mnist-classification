# FashionMNIST Image Classification

A PyTorch-based computer vision project that classifies grayscale clothing images into 10 distinct categories using the FashionMNIST dataset.

---

## Overview

The dataset consists of $28 \times 28$ grayscale images across 10 clothing categories, split into:
* **Training set:** 50,000 images (used to optimize model weights)
* **Validation set:** 10,000 images (used for hyperparameter tuning, model selection, and monitoring training)
* **Test set:** 10,000 images (held out for final unbiased evaluation)

> **Note on Methodology:** The test set remains strictly untouched during development and will only be evaluated once final hyperparameters (learning rate, number of epochs, etc.) are established.

---

## Dataset & Class Labels

| Label | Description |
| :---: | :--- |
| **0** | T-shirt/top |
| **1** | Trouser |
| **2** | Pullover |
| **3** | Dress |
| **4** | Coat |
| **5** | Sandal |
| **6** | Shirt |
| **7** | Sneaker |
| **8** | Bag |
| **9** | Ankle boot |

![FashionMNIST Sample Classes](class_image.png)

---

## Prerequisites & Installation

Ensure you have Python installed along with the required dependencies:

```bash
pip install torch torchvision matplotlib
```

---

## Usage

Clone the repository and navigate to the project directory

```bash
git clone https://github.com/prasun50/fashion-mnist-classification
cd fashion-mnist-classification
```

To train the baseline `LinearModel` with a specific learning rate and epoch count:

```bash
python src/train.py --model LinearModel -lr 0.05 --max-epoches 5
```

To display misclassified images during training or evaluation, pass the `-v` / `--visualize-wrong-preds` flag:

```shell
python src/train.py --model=LinearModel -lr=0.1 --max-epoches=3 -v
```

---

## Model Architectures & Results

| Model Architecture | Parameters | Validation Loss | Validation Accuracy |
| :--- | :--- | :--- | :--- |
| Baseline (Linear) | ~7.9K | 0.4961 | 83.42% |
| LeNet | ~6.2k | 0.35 | 87.16 |

---

## Project Structure

```text
|-- models/               # Neural network architecture definitions
|-- notebooks/            # Exploratory analysis & experiments
|-- src/
|   |-- train.py          # Training pipeline
|   |-- utils.py          # Data loaders and helper functions
|
|-- .gitignore            
|-- README.md             
|-- class_image.png       # image of all the classes
```