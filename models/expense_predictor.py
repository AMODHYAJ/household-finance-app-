import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

class ExpensePredictor:
    def __init__(self, model_path="models/expense_predictor.pkl"):
        self.model_path = model_path
        self.model = None

    def train(self, df):
        df = df[df["type"] == "expense"].copy()
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
        monthly = df.groupby(["year", "month"])["amount"].sum().reset_index()

        X = monthly[["year", "month"]]
        y = monthly["amount"]
        self.model = LinearRegression().fit(X, y)

        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)

    def predict_next_month(self, month, year):
        if self.model is None:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
        # Use DataFrame with feature names to avoid warning
        X_pred = pd.DataFrame([[year, month]], columns=["year", "month"])
        return self.model.predict(X_pred)[0]

    def evaluate(self, test_df):
        test_df = test_df[test_df["type"] == "expense"].copy()
        test_df["date"] = pd.to_datetime(test_df["date"])
        test_df["month"] = test_df["date"].dt.month
        test_df["year"] = test_df["date"].dt.year
        monthly = test_df.groupby(["year", "month"])["amount"].sum().reset_index()

        if self.model is None:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)

        X_test = monthly[["year", "month"]]
        y_test = monthly["amount"]
        # Use DataFrame with feature names to avoid warning
        preds = self.model.predict(X_test)
        return np.mean(np.abs(preds - y_test))  # MAE
