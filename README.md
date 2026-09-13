# 🏦 Home Loan Default Prediction using Deep Learning

A deep learning project to predict the probability of loan default using historical home loan application data.

This project was developed as a **Course-End Project for Deep Learning with Keras and TensorFlow** and focuses on handling an imbalanced financial dataset and building a neural-network-based classification model.

---

## 📌 Project Overview

For a safe and secure lending experience, analyzing historical loan data can help identify applicants who may be at higher risk of default.

The objective of this project is to build a **deep learning classification model** that predicts whether a loan applicant is likely to repay the loan or default based on historical applicant information.

### Domain

**Finance / Banking**

### Problem Type

**Binary Classification**

* `0` → Loan Repaid / Non-Default
* `1` → Loan Default

---

## 🎯 Objectives

The main objectives of this project are:

* Load and explore the loan dataset
* Identify missing/null values
* Analyze the distribution of the target variable
* Handle the highly imbalanced dataset
* Visualize class distribution before and after balancing
* Handle missing numerical and categorical values
* Encode categorical variables
* Scale the input features
* Build a deep learning classification model
* Evaluate the model using Sensitivity (Recall)
* Evaluate model performance using ROC-AUC
* Analyze model predictions using a confusion matrix

---

## 📊 Dataset

The dataset contains historical home loan application information.

Dataset is huge So I have uploaded the dataset in G-Drive, you can find the link in the dataset.text file.

The target variable is:

```text
TARGET
```

where:

```text
0 = Repaid / Non-Default
1 = Default
```

The dataset is highly imbalanced, with approximately:

* **92%** non-default/repaid applications
* **8%** default applications

Because of this imbalance, accuracy alone is not sufficient to evaluate the model.

> **Note:** If the dataset is not included in this repository due to size or redistribution restrictions, place `loan_data.csv` in the project directory before running the notebook.

---

## 🔄 Project Workflow

The project follows this workflow:

```text
Raw Loan Data
      ↓
Data Loading
      ↓
Data Exploration
      ↓
Missing Value Analysis
      ↓
Target Distribution Analysis
      ↓
Train/Test Split
      ↓
Missing Value Imputation
      ↓
Categorical Encoding
      ↓
Feature Scaling
      ↓
Training Data Balancing
      ↓
Deep Learning Model
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Sensitivity + ROC-AUC
```

---

## 🧹 Data Preprocessing

### 1. ID Removal

The `SK_ID_CURR` identifier column is removed because it is an ID rather than a meaningful predictive feature.

### 2. Train/Test Split

The dataset is divided into:

* 80% training data
* 20% testing data

Stratified splitting is used to preserve the class distribution.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### 3. Missing Value Handling

Missing numerical values are handled using **median imputation**.

Missing categorical values are handled using the **most-frequent value**.

```text
Numerical → Median
Categorical → Most Frequent
```

### 4. Categorical Encoding

Categorical features are converted into numerical features using:

**One-Hot Encoding**

```python
OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)
```

### 5. Feature Scaling

The processed features are standardized using:

```python
StandardScaler()
```

---

## ⚖️ Handling Class Imbalance

The dataset contains significantly fewer default cases than non-default cases.

To address this problem, **Random Oversampling** is applied to the training data.

```python
RandomOverSampler(random_state=42)
```

The test dataset is **not balanced**.

This is important because the model should be evaluated on the original class distribution rather than an artificially balanced test set.

---

## 🧠 Deep Learning Model

A feedforward neural network is built using **Keras and TensorFlow**.

### Model Architecture

```text
Input Layer
     ↓
Dense Layer – 128 neurons – ReLU
     ↓
Dropout – 30%
     ↓
Dense Layer – 64 neurons – ReLU
     ↓
Dropout – 20%
     ↓
Dense Layer – 32 neurons – ReLU
     ↓
Output Layer – 1 neuron – Sigmoid
```

### Configuration

**Optimizer:**

```text
Adam
```

**Loss Function:**

```text
Binary Crossentropy
```

**Evaluation Metrics:**

```text
Accuracy
Sensitivity / Recall
```

---

## 🛑 Early Stopping

Early stopping is used during training to help prevent unnecessary training and overfitting.

The model monitors:

```text
Validation Loss
```

and restores the best model weights.

---

## 📈 Model Evaluation

Because this is an imbalanced classification problem, several metrics are considered.

### Accuracy

Measures the overall percentage of correct predictions.

However, accuracy alone can be misleading when the classes are highly imbalanced.

### Precision

Measures how many applicants predicted as defaults were actually defaults.

### Sensitivity / Recall

Sensitivity is particularly important in this project because it measures how many actual defaulters were correctly identified.

```text
Sensitivity = True Positives / (True Positives + False Negatives)
```

### F1 Score

The F1 score provides a balance between precision and recall.

### ROC-AUC

ROC-AUC measures how well the model distinguishes between the two classes across different classification thresholds.

### Confusion Matrix

The confusion matrix provides a detailed view of:

* True Positives
* True Negatives
* False Positives
* False Negatives

---

## 📊 Visualizations

The notebook includes visualizations for:

* Target class distribution
* Training data before balancing
* Training data after balancing
* Training and validation loss
* Training and validation sensitivity
* Confusion matrix
* ROC curve

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* TensorFlow
* Keras
* Jupyter Notebook

---

## 📁 Project Files

```text
home-loan-default-prediction-deep-learning/
│
├── home_loan_corrected.ipynb    # Main project notebook
├── loan_data.csv                # Loan dataset (if permitted)
├── README.md                    # Project documentation

```

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/home-loan-default-prediction-deep-learning.git
```

### 2. Navigate to the project

```bash
cd home-loan-default-prediction-deep-learning
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place the dataset in the project directory with the filename:

```text
loan_data.csv
```

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
home_loan_corrected.ipynb
```

and run the cells sequentially.

---

## 📌 Key Learning Outcomes

Through this project, the following concepts were applied:

* Exploratory data analysis
* Missing-value treatment
* Train/test splitting
* Stratified sampling
* Categorical feature encoding
* Feature scaling
* Imbalanced classification
* Random oversampling
* Artificial neural networks
* Keras and TensorFlow
* Binary classification
* Sensitivity / Recall
* Precision
* F1 Score
* Confusion Matrix
* ROC Curve
* ROC-AUC
* Early stopping

---

## ⭐ Project Highlights

* Built a complete deep learning classification workflow
* Addressed a highly imbalanced financial dataset
* Used Random Oversampling for the minority class
* Implemented a neural network using TensorFlow/Keras
* Evaluated the model using Sensitivity and ROC-AUC
* Included multiple visualizations for model and data analysis
