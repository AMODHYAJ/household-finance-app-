import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

class CategoryClassifier:
    def __init__(self, model_path="models/category_classifier.pkl"):
        self.model_path = model_path
        self.pipeline = None

    def train(self, df):
        if "note" not in df.columns:
            return
        df = df.dropna(subset=["note", "category"])
        X, y = df["note"], df["category"]
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", MultinomialNB())
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
        if "note" not in test_df.columns:
            return 0
        test_df = test_df.dropna(subset=["note", "category"])
        X, y = test_df["note"], test_df["category"]

        if self.pipeline is None:
            with open(self.model_path, "rb") as f:
                self.pipeline = pickle.load(f)

        preds = self.pipeline.predict(X)
        return accuracy_score(y, preds)
