import pandas as pd
import pickle
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    def __init__(self, model_path="models/anomaly_detector.pkl"):
        self.model_path = model_path
        self.model = None

    def train(self, df):
        expenses = df[df["type"] == "expense"].copy()
        X = expenses[["amount"]]
        self.model = IsolationForest(contamination=0.05, random_state=42)
        self.model.fit(X)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)

    def detect(self, df):
        expenses = df[df["type"] == "expense"].copy()
        if expenses.empty:
            return pd.DataFrame()

        if self.model is None:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)

        X = expenses[["amount"]]
        expenses["anomaly"] = self.model.predict(X)
        return expenses[expenses["anomaly"] == -1]  # anomalies

