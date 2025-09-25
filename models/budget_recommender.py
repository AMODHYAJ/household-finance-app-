import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class BudgetRecommender:
    def __init__(self, model_path="models/budget_recommender.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = None

    def train(self, df):
        expenses = df[df["type"] == "expense"].copy()
        if expenses.empty:
            return

        # Monthly category spending profile
        expenses["date"] = pd.to_datetime(expenses["date"])
        expenses["month"] = expenses["date"].dt.to_period("M")
        pivot = expenses.pivot_table(
            index="month",
            columns="category",
            values="amount",
            aggfunc="sum",
            fill_value=0
        )

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(pivot)

        self.model = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.model.fit(X_scaled)

        with open(self.model_path, "wb") as f:
            pickle.dump((self.model, self.scaler), f)

    def recommend(self, df):
        if df is None or df.empty:
            return ["No data for budget recommendations."]

        if self.model is None or self.scaler is None:
            with open(self.model_path, "rb") as f:
                self.model, self.scaler = pickle.load(f)

        expenses = df[df["type"] == "expense"].copy()
        if expenses.empty:
            return ["No expenses found."]

        expenses["date"] = pd.to_datetime(expenses["date"])
        last_month = expenses["date"].dt.to_period("M").max()
        pivot = expenses[expenses["date"].dt.to_period("M") == last_month].copy()
        pivot["month"] = pivot["date"].dt.to_period("M")
        pivot = pivot.pivot_table(
            index="month",
            columns="category",
            values="amount",
            aggfunc="sum",
            fill_value=0
        )


        X_scaled = self.scaler.transform(pivot)
        cluster = self.model.predict(X_scaled)[0]

        recs = []
        if cluster == 0:
            recs.append("💡 You spend heavily in one category. Set a cap for it to free up savings.")
        elif cluster == 1:
            recs.append("✅ Your spending is balanced across categories. Maintain at least 20% savings.")
        elif cluster == 2:
            recs.append("⚠ You have irregular big expenses. Build an emergency fund to stay safe.")

        return recs
