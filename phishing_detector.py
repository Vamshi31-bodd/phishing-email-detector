
import pandas as pd
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. LOAD AND CLEAN DATASET
# ==========================================

print("=" * 50)
print("       PHISHING EMAIL DETECTION MODEL")
print("=" * 50)

dataset_path = Path("dataset/emails.csv")

if not dataset_path.is_file():
    raise FileNotFoundError(
        "Dataset not found. Check dataset/emails.csv"
    )

df = pd.read_csv(dataset_path)

df = df[["Email Text", "Email Type"]].copy()

df.dropna(
    subset=["Email Text", "Email Type"],
    inplace=True
)

df["Email Text"] = df["Email Text"].astype(str).str.strip()
df["Email Type"] = df["Email Type"].astype(str).str.strip()

df.drop_duplicates(inplace=True)

# Convert email labels into numerical values
df["label"] = df["Email Type"].map({
    "Phishing Email": 1,
    "Safe Email": 0
})

# Remove records with unexpected labels
df.dropna(subset=["label"], inplace=True)
df["label"] = df["label"].astype(int)

print("\nDataset loaded successfully!")
print("Total usable emails:", len(df))

print("\nEmail Distribution:")
print(df["Email Type"].value_counts())


# ==========================================
# 2. EMAIL FEATURE EXTRACTION
# ==========================================

def extract_features(email):

    email = str(email)
    email_lower = email.lower()

    suspicious_keywords = [
        "urgent",
        "verify",
        "password",
        "account",
        "winner",
        "prize",
        "click here",
        "bank",
        "suspended",
        "confirm",
        "free",
        "login",
        "update",
        "security",
        "payment"
    ]

    url_count = len(
        re.findall(
            r"https?://|www\.",
            email_lower
        )
    )

    keyword_count = sum(
        email_lower.count(word)
        for word in suspicious_keywords
    )

    exclamation_count = email.count("!")

    email_length = len(email)

    word_count = len(email.split())

    return {
        "url_count": url_count,
        "keyword_count": keyword_count,
        "exclamation_count": exclamation_count,
        "email_length": email_length,
        "word_count": word_count
    }


features = pd.DataFrame(
    df["Email Text"].apply(extract_features).tolist(),
    index=df.index
)

print("\nEmail features extracted successfully!")


# ==========================================
# 3. PREPARE TRAINING DATA
# ==========================================

X = pd.concat(
    [
        df[["Email Text"]],
        features
    ],
    axis=1
)

y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 4. TEXT AND NUMERICAL PREPROCESSING
# ==========================================

feature_columns = [
    "url_count",
    "keyword_count",
    "exclamation_count",
    "email_length",
    "word_count"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "text",
            TfidfVectorizer(
                max_features=5000,
                stop_words="english"
            ),
            "Email Text"
        ),
        (
            "numeric",
            StandardScaler(),
            feature_columns
        )
    ]
)


# ==========================================
# 5. CREATE MACHINE LEARNING PIPELINE
# ==========================================

model = Pipeline([
    ("preprocessor", preprocessor),

    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])


# ==========================================
# 6. TRAIN THE MODEL
# ==========================================

print("\n========== MODEL TRAINING ==========")

model.fit(X_train, y_train)

print("Model training completed successfully!")


# ==========================================
# 7. MODEL EVALUATION
# ==========================================

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n========== MODEL PERFORMANCE ==========")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Safe Email",
            "Phishing Email"
        ]
    )
)


# ==========================================
# 8. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

print("\nConfusion Matrix:")
print(cm)

# Create confusion matrix visualization
plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Safe", "Phishing"],
    yticklabels=["Safe", "Phishing"]
)

plt.title("Phishing Email Detection - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()


# ==========================================
# 9. SAVE CONFUSION MATRIX IMAGE
# ==========================================

reports_dir = Path("reports")

if reports_dir.exists() and not reports_dir.is_dir():
    raise NotADirectoryError(
        "'reports' exists as a file. "
        "Rename it or create a reports folder."
    )

reports_dir.mkdir(
    parents=True,
    exist_ok=True
)

matrix_path = reports_dir / "confusion_matrix.png"

plt.savefig(
    matrix_path,
    dpi=300
)

plt.show()

print("\nConfusion matrix saved:", matrix_path)


# ==========================================
# 10. SAVE TRAINED MODEL
# ==========================================

models_dir = Path("models")

if models_dir.exists() and not models_dir.is_dir():

    print("\n[WARNING] 'models' exists as a file.")
    print("Saving the trained model in the project root instead.")

    model_path = Path("phishing_model.pkl")

else:

    models_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = models_dir / "phishing_model.pkl"


joblib.dump(model,model_path
)

print("\nModel saved successfully!")
print("Model location:", model_path)


# ==========================================
# 11. FINAL SUMMARY
# ==========================================

print("\n========== FINAL PROJECT SUMMARY ==========")

print("Total Dataset:", len(df))

print("Training Samples:", len(X_train))

print("Testing Samples:", len(X_test))

print(
    "Model Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("Model: Logistic Regression")

print("Text Feature Extraction: TF-IDF")

print("Numerical Features: 5")

print("Confusion Matrix: Generated")

print("Trained Model: Saved")

print("\nPHISHING EMAIL MODEL COMPLETED!")

print("=" * 50)
