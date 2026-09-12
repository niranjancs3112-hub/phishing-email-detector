from flask import Flask, render_template, request
import joblib
import re
from scipy.sparse import hstack, csr_matrix


app = Flask(__name__)


# ==============================
# PROJECT INFORMATION
# ==============================

MODEL_ACCURACY = 98.58

TOTAL_EMAILS = 82485

PHISHING_EMAILS = 42891

SAFE_EMAILS = 39595


# ==============================
# CONFUSION MATRIX
# ==============================

CONFUSION_MATRIX = {
    "true_safe": 7783,
    "false_phishing": 136,
    "false_safe": 99,
    "true_phishing": 8479
}


# ==============================
# PREDICTION HISTORY
# ==============================

prediction_history = []
phishing_predictions = 0
safe_predictions = 0

# ==============================
# LOAD TRAINED MODEL
# ==============================

model = joblib.load("phishing_model.pkl")

vectorizer = joblib.load("tfidf_vectorizer.pkl")


# ==============================
# SUSPICIOUS KEYWORDS
# ==============================

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


# ==============================
# CLEAN TEXT
# ==============================

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(r'<.*?>', ' ', text)

    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# ==============================
# GET EXTRA FEATURES
# ==============================

def get_features(text):

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

    return len(urls), keyword_count


# ==============================
# HOME PAGE
# ==============================

@app.route("/", methods=["GET", "POST"])
def home():
    global phishing_predictions, safe_predictions
    result = None

    url_count = 0

    keyword_count = 0

    confidence = 0


    # Email submitted
    if request.method == "POST":

        email = request.form["email"]


        # Get URL and keyword features
        url_count, keyword_count = get_features(email)


        # Clean email
        cleaned_email = clean_text(email)


        # TF-IDF
        tfidf = vectorizer.transform(
            [cleaned_email]
        )


        # Extra features
        extra_features = csr_matrix(
            [[url_count, keyword_count]]
        )


        # Combine features
        final_features = hstack([
            tfidf,
            extra_features
        ])


        # Make prediction
        prediction = model.predict(
            final_features
        )[0]


        # Get probabilities
        probabilities = model.predict_proba(
            final_features
        )[0]


        # Calculate confidence
        confidence = round(
            max(probabilities) * 100,
            2
        )


        # ==========================
        # RESULT
        # ==========================

        if prediction == 1:
            result = "PHISHING"
            phishing_predictions += 1
        else:
            result = "SAFE"
            safe_predictions += 1

        # ==========================
        # SAVE PREDICTION HISTORY
        # ==========================

        prediction_history.append({
    "email": email,
    "result": result,
    "confidence": confidence
})

    # ==============================
    # TOTAL PREDICTIONS
    # ==============================

    total_predictions = len(
        prediction_history
    )


    # ==============================
    # SEND DATA TO HTML
    # ==============================

    return render_template(
        "index.html",

        result=result,

        url_count=url_count,

        keyword_count=keyword_count,

        confidence=confidence,

        accuracy=MODEL_ACCURACY,

        total_emails=TOTAL_EMAILS,

        phishing_emails=PHISHING_EMAILS,

        safe_emails=SAFE_EMAILS,

        total_predictions=total_predictions,
        phishing_predictions=phishing_predictions,
        safe_predictions=safe_predictions,

        history=prediction_history,

        confusion_matrix=CONFUSION_MATRIX
    )


# ==============================
# CLEAR HISTORY
# ==============================

@app.route("/clear-history")
def clear_history():
    global phishing_predictions, safe_predictions
    prediction_history.clear()
    phishing_predictions = 0
    safe_predictions = 0



    # Total predictions becomes 0
    total_predictions = len(
        prediction_history
    )


    return render_template(
        "index.html",

        result=None,

        url_count=0,

        keyword_count=0,

        confidence=0,

        accuracy=MODEL_ACCURACY,

        total_emails=TOTAL_EMAILS,

        phishing_emails=PHISHING_EMAILS,

        safe_emails=SAFE_EMAILS,

        total_predictions=total_predictions,
        phishing_predictions=phishing_predictions,
        safe_predictions=safe_predictions,

        history=prediction_history,

        confusion_matrix=CONFUSION_MATRIX
    )


# ==============================
# RUN FLASK
# ==============================

if __name__ == "__main__":

    app.run(debug=True)