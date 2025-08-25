from flask import Flask, render_template, request
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# Constants
MODEL_PATH = "LinearSVCTuned.pkl"
TFIDF_CANDIDATES = ["tfidfmodel.pkl", "tfidfvectorizer.pkl", "tfidfvectoizer.pkl"]


def load_stopwords(path="stopwords.txt"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().splitlines()
    return None


def load_vectorizer(stopwords=None):
    for fn in TFIDF_CANDIDATES:
        if os.path.exists(fn):
            try:
                with open(fn, "rb") as f:
                    obj = pickle.load(f)
                # If the pickle contains a fitted TfidfVectorizer instance
                if hasattr(obj, "transform") and hasattr(obj, "vocabulary_"):
                    return obj
                # If the pickle contains a vocabulary dict
                if isinstance(obj, dict):
                    return TfidfVectorizer(stop_words=stopwords, lowercase=True, vocabulary=obj)
            except Exception:
                continue

    # Fallback: create an unfitted vectorizer (transform will fail until fitted)
    print("Warning: no fitted TF-IDF vectorizer found. Using a fresh TfidfVectorizer.")
    return TfidfVectorizer(stop_words=stopwords, lowercase=True)


def load_model(path=MODEL_PATH):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading model {path}: {e}")
    else:
        print(f"Model file {path} not found.")
    return None


def create_app():
    app = Flask(__name__, static_folder="static", template_folder="templates")

    stopwords = load_stopwords()
    vectorizer = load_vectorizer(stopwords)
    model = load_model()

    @app.route("/", methods=["GET", "POST"])
    def index():
        label = None
        score = None
        error = None

        if request.method == "POST":
            text = request.form.get("text", "").strip()
            if not text:
                error = "Please enter some text to classify."
            else:
                try:
                    X = vectorizer.transform([text])
                except Exception as e:
                    X = None
                    error = f"Vectorization error: {e}"

                if X is not None:
                    if model is None:
                        error = "Model not available on server."
                    else:
                        try:
                            pred = model.predict(X)[0]
                            label = "Bullying" if int(pred) == 1 else "Non-bullying"
                            # Confidence/probability when available
                            if hasattr(model, "predict_proba"):
                                prob = model.predict_proba(X)[0]
                                score = float(prob[int(pred)])
                            elif hasattr(model, "decision_function"):
                                df = model.decision_function(X)
                                # convert decision score to a bounded 0..1-ish value
                                score = float(1 / (1 + abs(df[0])))
                            else:
                                score = None
                        except Exception as e:
                            error = f"Prediction error: {e}"

        return render_template("index.html", label=label, score=score, error=error)

    @app.route("/health", methods=["GET"])
    def health():
        # health endpoint: returns 200 when model and vectorizer are loaded
        ok = True
        details = {}
        if model is None:
            ok = False
            details['model'] = 'missing'
        else:
            details['model'] = 'loaded'
        if vectorizer is None:
            ok = False
            details['vectorizer'] = 'missing'
        else:
            details['vectorizer'] = 'loaded'
        status_code = 200 if ok else 503
        return (details, status_code)

    return app


if __name__ == "__main__":
    create_app().run(debug=True, host="127.0.0.1", port=5000)
