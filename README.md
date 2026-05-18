
# **Online-Payment-Fraud-Prediction-Using-ML**

## Project Overview
This project implements a fraud detection system using a Random Forest Classifier. It processes transactional data, performs feature engineering, and provides an interactive interface for predicting whether a new transaction is fraudulent or legitimate. The system also incorporates rule-based anomaly detection to enhance its prediction capabilities.

## Features
- **Data Loading and Cleaning**: Handles `fraud_legit_10k_30_70_ratio.csv` data, drops unnecessary columns, and addresses potential outliers.
- **Feature Engineering**: Creates new features such as `sender_balance_error`, `receiver_balance_error`, `hour_of_day`, `amount_to_balance_ratio`, `empties_account`, and `is_large_transaction` to improve model performance.
- **Categorical Data Encoding**: Transforms categorical transaction types into numerical representations.
- **Advanced Exploratory Data Analysis (EDA)**: Visualizes transaction distributions, fraud percentages, amount comparisons, and correlations to understand the dataset better. Includes a 2D PCA visualization to show data separation.
- **Random Forest Classifier**: Trains a robust Random Forest model for fraud prediction.
- **Model Evaluation**: Assesses model performance using accuracy, confusion matrix, classification report, cross-validation, and ROC curve/AUC score.
- **Interactive Prediction**: Allows users to input transaction details and receive a real-time fraud prediction with a probability score and explanation.
- **Hybrid Anomaly Detection**: Combines the machine learning model's predictions with rule-based logic to flag transactions with significant balance discrepancies.

## Local Setup
To set up and run this project on your local machine, follow these steps:

### Prerequisites
- Python 3.7+
- pip (Python package installer)

### 1. Clone the Repository
First, clone this GitHub repository to your local machine:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

### 2. Install Dependencies
Navigate into the project directory and install the required Python packages using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Obtain the Dataset
The project relies on a CSV file named `fraud_legit_10k_30_70_ratio.csv`. You must place this file in the same directory as the Python script. If you do not have this file, the script will output a `FileNotFoundError`.

*(Note: You will need to provide instructions on how users can obtain this dataset, e.g., a link to download it, or if it should be included in the repository itself.)*

### 4. Run the Script
There are two main scripts in the notebook (`Ld3HpS4-jg6p` for without PCA and `uNtbDadO-_on` for with PCA). You can run the desired Python script from your terminal:

```bash
python your_script_name_without_pca.py # Assuming you save the first code cell as a .py file
```
Or if you're using a Jupyter/Colab environment, simply execute the cells in order.

## Usage
Once the script is running, it will perform data loading, cleaning, feature engineering, model training, and evaluation. After these initial steps, it will prompt you to enter details for a new transaction under the `USER INPUT PREDICTION` section.

Follow the prompts to enter the required information:
- `Step (hour)`
- `Transaction Type` (e.g., CASH_OUT, TRANSFER)
- `Amount`
- `Sender Old Balance`
- `Sender New Balance`
- `Receiver Old Balance`
- `Receiver New Balance`

The script will then output a prediction (Fraudulent or Legit), a fraud risk probability, and an explanation for the prediction.

## Model Details
The core of the fraud detection system is a Random Forest Classifier, trained on scaled features. The model incorporates a `class_weight='balanced_subsample'` to handle imbalanced datasets effectively. Feature scaling is performed using `StandardScaler` to normalize numerical features.

For the version including PCA (`uNtbDadO-_on`), Principal Component Analysis is used primarily for dimensionality reduction for visualization purposes, helping to understand the separability of fraudulent and legitimate transactions in a 2D space.

## Results and Evaluation
The model's performance is evaluated using standard metrics:
- **Accuracy Score**: Overall correctness of predictions.
- **Confusion Matrix**: Visual representation of true positives, true negatives, false positives, and false negatives.
- **Classification Report**: Precision, recall, and F1-score for each class.
- **Cross-Validation**: 5-Fold Stratified Cross-Validation on the training set to ensure model robustness.
- **ROC Curve and AUC Score**: Measures the model's ability to distinguish between fraud and legitimate transactions across various thresholds.

## Contributing
Feel free to fork this repository, submit pull requests, or open issues to suggest improvements or report bugs.
