import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import numpy as np

class CategoryClassifier:
    def __init__(self, model_path="models/category_classifier.pkl"):
        self.model_path = model_path
        self.pipeline = None

    def train(self, df):
        if "note" not in df.columns or "category" not in df.columns:
            return
        df = df.dropna(subset=["note", "category"])
        if df.empty:
            return

        X, y = df["note"], df["category"]
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, solver="lbfgs", multi_class='auto'))
        ])
        self.pipeline.fit(X, y)

        with open(self.model_path, "wb") as f:
            pickle.dump(self.pipeline, f)

    def predict(self, note_text):
        if self.pipeline is None:
            with open(self.model_path, "rb") as f:
                self.pipeline = pickle.load(f)
        return self.pipeline.predict([note_text])[0]

    def predict_top(self, note_text, top_n=3):
        """
        Returns top N categories with confidence scores
        as a list of tuples: [(category, confidence), ...]
        """
        if self.pipeline is None:
            with open(self.model_path, "rb") as f:
                self.pipeline = pickle.load(f)

        probs = self.pipeline.predict_proba([note_text])[0]
        classes = self.pipeline.classes_
        top_indices = np.argsort(probs)[::-1][:top_n]
        return [(classes[i], probs[i]) for i in top_indices]

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
