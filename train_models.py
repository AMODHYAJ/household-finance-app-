import pandas as pd
import os
from models.expense_predictor import ExpensePredictor
from models.anomaly_detector import AnomalyDetector
from models.category_classifier import CategoryClassifier
from models.budget_recommender import BudgetRecommender

DATA_PATH = "data/finance_dataset.csv"
TEST_PATH = "data/finance_test.csv"
MODEL_DIR = "models/"

def check_dataset_quality(df: pd.DataFrame, name: str):
    """Responsible AI check: fairness, transparency, and ethics in dataset handling."""
    print(f"\n🔎 Responsible AI Check for {name} dataset:")

    # Missing values check
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"⚠️ {missing} missing values detected. Consider cleaning before training.")
    else:
        print("✅ No missing values detected.")

    # Fairness check: category balance
    if "category" in df.columns:
        counts = df["category"].value_counts(normalize=True)
        if counts.max() > 0.8:  # >80% dominance
            print("⚠️ Category imbalance detected. Some categories may be underrepresented.")
        else:
            print("✅ Categories appear reasonably balanced.")

    # Sensitive data check
    if "note" in df.columns:
        print("ℹ️ Privacy note: Only masked text will be logged. Raw notes remain private.")

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Responsible AI dataset checks
    check_dataset_quality(df, "Training")
    check_dataset_quality(test_df, "Test")

    print("\n✅ Training with dataset:", len(df), "rows")

    # ---- Expense Predictor ----
    exp_pred = ExpensePredictor(model_path=os.path.join(MODEL_DIR, "expense_predictor.pkl"))
    exp_pred.train(df)
    print("📈 Expense Predictor MAE:", exp_pred.evaluate(test_df))
    print("ℹ️ Transparency: Predictions use RandomForest on past expense trends.")

    # ---- Anomaly Detector ----
    anomaly_det = AnomalyDetector(model_path=os.path.join(MODEL_DIR, "anomaly_detector.pkl"))
    anomaly_det.train(df)
    anomalies = anomaly_det.detect(test_df)
    print("🚨 Anomalies in test set:", len(anomalies))
    print("ℹ️ Transparency: Anomalies flagged using Isolation Forest on spending patterns.")

    # ---- Category Classifier ----
    cat_clf = CategoryClassifier(model_path=os.path.join(MODEL_DIR, "category_classifier.pkl"))
    cat_clf.train(df)
    print("📝 Category Classifier Accuracy:", cat_clf.evaluate(test_df))
    print("ℹ️ Transparency: Categories are predicted using TF-IDF + Logistic Regression.")

    # ---- Budget Recommender ----
    budget_rec = BudgetRecommender(model_path=os.path.join(MODEL_DIR, "budget_recommender.pkl"))
    budget_rec.train(df)
    recs = budget_rec.recommend(test_df)
    print("💡 Budget Recommendations Sample:", recs[:3])  # show first 3 recommendations
    print("ℹ️ Transparency: Recommendations are cluster-based comparisons of your spending.")

    print("\n✅ All models trained & saved to 'models/' with Responsible AI checks included.")

if __name__ == "__main__":
    main()
