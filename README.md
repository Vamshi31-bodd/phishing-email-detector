
# Phishing Email Detection Model

## Project Overview

The Phishing Email Detection Model is a Machine Learning project developed using Python and Scikit-learn. It identifies whether an email is legitimate or potentially malicious by analyzing email text and extracting important features.

The model uses Natural Language Processing (NLP) and Logistic Regression to classify emails into two categories: Safe Email and Phishing Email.

## Objectives

- Detect phishing emails using Machine Learning.
- Analyze email content and suspicious URLs.
- Extract important textual and numerical features.
- Classify emails as Safe or Phishing.
- Evaluate model performance using accuracy and a confusion matrix.
- Provide predictions for user-entered emails.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Matplotlib
- Seaborn
- Joblib

## Dataset

The project uses a dataset containing legitimate and phishing emails.

- Original dataset size: 18,650 emails
- Usable emails after preprocessing: 17,524
- Safe emails: 10,979
- Phishing emails: 6,545

## Features Extracted

The model analyzes the following features:

1. URL count
2. Phishing keyword count
3. Exclamation mark count
4. Email length
5. Word count
6. TF-IDF text features

## Machine Learning Model

Algorithm: Logistic Regression

Text Feature Extraction: TF-IDF

Training and Testing Split:
- Training: 80%
- Testing: 20%

## Model Performance

Accuracy: 97.23%

Confusion Matrix:

| Actual / Predicted | Safe | Phishing |
|---|---:|---:|
| Safe | 2166 | 30 |
| Phishing | 67 | 1242 |

## Project Structure

phishing-email-detector/
- dataset/
- models/
- reports/
- phishing_detector.py
- predict_email.py
- requirements.txt
- .gitignore
- README.md

## How to Run the Project

### 1. Install dependencies

pip install -r requirements.txt

### 2. Train the model

python phishing_detector.py

### 3. Run live email prediction

python predict_email.py

### 4. Enter email text

Paste the email content when prompted. The model will classify it as Safe Email or Phishing Email and display its confidence scores.

## Model Output

The model provides:
- Email classification
- Prediction confidence scores
- Confusion matrix
- Model accuracy and classification report

## Disclaimer

This project is developed for educational and cybersecurity awareness purposes. Predictions are automated and should not be treated as a guarantee that an email is safe or malicious.
