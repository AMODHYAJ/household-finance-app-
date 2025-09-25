import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

class CategoryClassifier:
    def __init__(self, model_path="models/category_classifier.pkl"):
        self.model_path = model_path
        self.pipeline = None

    def check_data_balance(self, df):
        """Responsible AI check for fairness: warn if data is imbalanced."""
        if "category" in df.columns:
            counts = df["category"].value_counts(normalize=True)
            if counts.max() > 0.8:
                print("⚠️ Responsible AI Check: Training data heavily imbalanced. Some categories may be underrepresented.")

    def train(self, df):
        if "note" not in df.columns or "category" not in df.columns:
            return
        df = df.dropna(subset=["note", "category"])
        if df.empty:
            return

        # Fairness check
        self.check_data_balance(df)

        X, y = df["note"], df["category"]
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, solver="lbfgs"))
        ])
        self.pipeline.fit(X, y)

        with open(self.model_path, "wb") as f:
            pickle.dump(self.pipeline, f)

    def predict(self, note_text):
        if self.pipeline is None:
            with open(self.model_path, "rb") as f:
                self.pipeline = pickle.load(f)
        return self.pipeline.predict([note_text])[0]

    def evaluate(self, test_df):
        if "note" not in test_df.columns or "category" not in test_df.columns:
            return 0
        test_df = test_df.dropna(subset=["note", "category"])
        if test_df.empty:
            return 0

        X, y = test_df["note"], test_df["category"]

        if self.pipeline is None:
            with open(self.model_path, "rb") as f:
                self.pipeline = pickle.load(f)

        preds = self.pipeline.predict(X)
        return accuracy_score(y, preds)