# Language Identification with a Neural Network Implemented from Scratch

Authors: **Beso Mikaberidze**, **Guram Mikaberidze**

This project implements a multiclass language identification system based on a deep feed-forward neural network, built entirely from scratch without using any deep learning frameworks (e.g., PyTorch, TensorFlow, Keras).

This project is not intended as a production-ready language identifier, but as a deep dive into neural network mechanics, such as forward pass, backpropagation, optimization, and regularization.

## Contents
- [Overview](#overview)
  - [Problem Formulation](#problem-formulation)
  - [Dataset and Preprocessing](#dataset-and-preprocessing)
  - [Model Architecture and Training](#model-architecture-and-training)
  - [Experimental Setup and Results](#experimental-setup-and-results)
- [Usage](#usage)
- [Contact](#contact)

---

## Overviev

### Problem Formulation

- **Task:** Multiclass language identification  
- **Input:** A single sentence represented as a fixed-length character n-gram frequency vector
- **Output:** One of *N* language labels  
- **Model type:** Fully connected neural network (MLP)  


### Dataset and Preprocessing

The model is trained using sentence-level data from the publicly available [**Tatoeba sentence dataset**](https://downloads.tatoeba.org/exports/sentences.csv).

Preprocessing consists of the following steps:

1. **Dataset filtering and splitting**
   - Selection of a fixed number of languages
   - Filtering by sentence length
   - Balancing the number of samples per language
   - Split into train, validation, and test sets

2. **Feature extraction**
   - Construction of a character n-gram vocabulary from the most frequent n-grams observed in each language  
   (ensuring that features capture language-specific orthographic patterns)
   - Transformation of sentences into vocabulary size n-gram frequency vectors,  
   followed by min–max normalization that uses training-set statistics


### Model Architecture and Training

The model is a **deep feed-forward neural network**, where:

- Hidden layers use **ReLU** activation
- The output layer uses **Softmax** for multiclass classification
- The loss function is **categorical cross-entropy**

#### Implemented from Scratch

No deep learning frameworks are used; all gradient computations and training logic are implemented manually:

- Forward pass
- Backpropagation
- Loss evaluation
- Gradient computation
- Parameter updates
- Optimization & Regularization
  - Adam optimizer
  - L2 regularization
  - Linear learning rate scheduling across epochs


### Experimental Setup and Results

- Number of languages: **6**  
- Evaluation metric: **Accuracy**  
- Test result: **98.12%**

Model selection is performed based on validation performance while tuning:
- number of hidden layers
- hidden layer sizes
- learning rate
- regularization strength

---

## Usage

### Setup

Clone the repository
```
git clone https://github.com/besom/language-id-mlp-from-scratch.git
cd language-id-mlp-from-scratch
```

Set up the virtual environment
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure and Run

All preprocessing, model, and training parameters are configurable via:
  ```
  config/config.json
  ```

Download the dataset (once) using:
  ```
  python -m src.scripts.download_dataset
  ```

Run the full preprocessing, training, and evaluation pipeline using:
  ```
  python -m src.scripts.run
  ```

Alternatively, experiments can be executed interactively via:
  ```
  notebooks/run.ipynb
  ```

---

## Contact

Beso Mikaberidze - beso.mikaberidze@gmail.com  
Guram Mikaberidze - guram.mikaberidze@gmail.com