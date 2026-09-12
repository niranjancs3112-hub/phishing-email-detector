import pandas as pd
import re
import joblib

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. LOAD DATASET
# ============================================================

data = pd.read_csv("phishing_email.csv")

# Rename the dataset column
data = data.rename(columns={
    "text_combined": "text"
})

print("========================================")
print("PHISHING EMAIL DETECTION")
print("========================================")

print("\nDataset loaded successfully!")
print("Total emails:", len(data))

print("\nLabel counts:")
print(data["label"].value_counts())


# ============================================================
# 2. REMOVE EMPTY EMAILS
# ============================================================

data["text"] = data["text"].fillna("")

data = data[data["text"].str.strip() != ""]

print("\nEmails after removing empty values:", len(data))


# ============================================================
# 3. KEEP ORIGINAL EMAIL TEXT
# ============================================================

data["raw_text"] = data["text"]


# ============================================================
# 4. CLEAN EMAIL TEXT
# ============================================================

def clean_text(text):

    text = str(text).lower()

    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


data["text"] = data["text"].apply(clean_text)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X = data["text"]
X_raw = data["raw_text"]
y = data["label"]

X_train, X_test, y_train, y_test, X_raw_train, X_raw_test = train_test_split(
    X,
    y,
    X_raw,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========================================")
print("DATA SPLIT")
print("========================================")

print("Training emails:", len(X_train))
print("Testing emails:", len(X_test))


# ============================================================
# 6. TF-IDF
# ============================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=20000
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF completed!")


# ============================================================
# 7. URL + SUSPICIOUS KEYWORD FEATURES
# ============================================================

suspicious_keywords = [
    "urgent",
    "verify",
    "login",
    "password",
    "account",
    "click",
    "bank",
    "confirm",
    "security",
    "winner",
    "prize",
    "claim",
    "suspended",
    "expire",
    "payment",
    "credit",
    "debit",
    "limited",
    "immediately"
]


def get_features(text):

    text = str(text)

    # Find URLs
    urls = re.findall(
        r'https?://\S+|www\.\S+',
        text,
        flags=re.IGNORECASE
    )

    # Count suspicious keywords
    keyword_count = sum(
        1
        for word in suspicious_keywords
        if re.search(
            r'\b' + re.escape(word) + r'\b',
            text,
            flags=re.IGNORECASE
        )
    )

    return [
        len(urls),
        keyword_count
    ]


print("\nExtracting URL and keyword features...")

train_extra = [
    get_features(text)
    for text in X_raw_train
]

test_extra = [
    get_features(text)
    for text in X_raw_test
]


# Convert extra features to sparse matrix
train_extra = csr_matrix(train_extra)

test_extra = csr_matrix(test_extra)


# ============================================================
# 8. COMBINE ALL FEATURES
# ============================================================

X_train_final = hstack([
    X_train_tfidf,
    train_extra
])

X_test_final = hstack([
    X_test_tfidf,
    test_extra
])

print("\n========================================")
print("FEATURES")
print("========================================")

print("TF-IDF + URL + Keyword features combined")

print("Final training shape:",
      X_train_final.shape)

print("Final testing shape:",
      X_test_final.shape)


# ============================================================
# 9. TRAIN LOGISTIC REGRESSION MODEL
# ============================================================

print("\n========================================")
print("MODEL TRAINING")
print("========================================")

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_final,
    y_train
)

print("Model training completed successfully!")
joblib.dump(model, "phishing_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model saved successfully!")


# ============================================================
# 10. MAKE TEST PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test_final
)

print("\nPredictions:")
print(y_pred)


# ============================================================
# 11. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n========================================")
print("MODEL PERFORMANCE")
print("========================================")

print(
    "Model Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Safe",
            "Phishing"
        ]
    )
)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=[
        "Safe",
        "Phishing"
    ],
    yticklabels=[
        "Safe",
        "Phishing"
    ]
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title(
    "Phishing Email Detection - Confusion Matrix"
)

plt.tight_layout()

plt.savefig("confusion_matrix.png")
plt.close()

print("\nConfusion matrix saved as confusion_matrix.png")


# ============================================================
# 14. TEST A NEW EMAIL
# ============================================================

print("\n========================================")
print("NEW EMAIL TEST")
print("========================================")

new_email = input(
    "\nEnter an email to check: "
)


# ------------------------------------------------------------
# Extract URL and keyword features BEFORE cleaning
# ------------------------------------------------------------

url_count, keyword_count = get_features(
    new_email
)

print("\nURLs found:", url_count)

print(
    "Suspicious keywords found:",
    keyword_count
)


# ------------------------------------------------------------
# Clean email
# ------------------------------------------------------------

new_email_clean = clean_text(
    new_email
)


# ------------------------------------------------------------
# Convert email to TF-IDF
# ------------------------------------------------------------

new_email_tfidf = vectorizer.transform(
    [new_email_clean]
)


# ------------------------------------------------------------
# Combine TF-IDF + URL + keyword features
# ------------------------------------------------------------

new_email_extra = csr_matrix([
    [url_count, keyword_count]
])


new_email_final = hstack([
    new_email_tfidf,
    new_email_extra
])


# ============================================================
# 15. PREDICT EMAIL
# ============================================================

prediction = model.predict(
    new_email_final
)[0]


# ============================================================
# 16. DISPLAY RESULT
# ============================================================

print("\n========================================")

if prediction == 1:

    print("Prediction: PHISHING")

    print(
        "Warning: This email appears suspicious."
    )

else:

    print("Prediction: SAFE")

    print(
        "This email appears to be legitimate."
    )

print("========================================")