
import re
import joblib
import pandas as pd
from pathlib import Path

# Load the trained model
model_path = Path("models") / "phishing_model.pkl"

if not model_path.exists():
    raise FileNotFoundError("Trained model not found. Run phishing_detector.py first.")

model = joblib.load(model_path)

# Same keywords used during model training
phishing_keywords = [
    "urgent", "verify", "password", "account",
    "winner", "prize", "click here", "bank",
    "suspended", "confirm", "free", "login",
    "update", "security", "payment"
]


def extract_features(email_text):

    text_lower = email_text.lower()

    url_count = len(
        re.findall(r"https?://|www\.", email_text, re.IGNORECASE)
    )

    keyword_count = sum(
        text_lower.count(keyword)
        for keyword in phishing_keywords
    )

    exclamation_count = email_text.count("!")

    email_length = len(email_text)

    word_count = len(re.findall(r"\b\w+\b", email_text))

    return {
        "Email Text": email_text,
        "url_count": url_count,
        "keyword_count": keyword_count,
        "exclamation_count": exclamation_count,
        "email_length": email_length,
        "word_count": word_count
    }


print("=" * 50)
print("       PHISHING EMAIL DETECTOR")
print("=" * 50)

email = input("\nPaste your email text here:\n\n")

if not email.strip():
    print("\nPlease enter some email text.")
else:

    features = extract_features(email)

    input_data = pd.DataFrame([features])

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    if prediction == 1:
        print("\nRESULT: PHISHING EMAIL")
        print("Warning: This email may be suspicious.")

    else:
        print("\nRESULT: SAFE EMAIL")
        print("No phishing indicators were detected by the model.")

    print("\nModel confidence scores:")

    for label, probability in zip(
        model.classes_, probabilities
    ):
        print(f"{label}: {probability * 100:.2f}%")

    print("\nNote: This is an automated prediction, not a guarantee.")
