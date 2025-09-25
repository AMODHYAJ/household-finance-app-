import pandas as pd
import os
from models.expense_predictor import ExpensePredictor
from models.anomaly_detector import AnomalyDetector
from models.category_classifier import CategoryClassifier
from models.budget_recommender import BudgetRecommender

DATA_PATH = "data/finance_dataset.csv"
TEST_PATH = "data/finance_test.csv"
MODEL_DIR = "models/"

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("✅ Training with dataset:", len(df), "rows")

    # ---- Expense Predictor ----
    exp_pred = ExpensePredictor(model_path=os.path.join(MODEL_DIR, "expense_predictor.pkl"))
    exp_pred.train(df)
    print("📈 Expense Predictor MAE:", exp_pred.evaluate(test_df))

    # ---- Anomaly Detector ----
    anomaly_det = AnomalyDetector(model_path=os.path.join(MODEL_DIR, "anomaly_detector.pkl"))
    anomaly_det.train(df)
    anomalies = anomaly_det.detect(test_df)
    print("🚨 Anomalies in test set:", len(anomalies))

    # ---- Category Classifier ----
    cat_clf = CategoryClassifier(model_path=os.path.join(MODEL_DIR, "category_classifier.pkl"))
    cat_clf.train(df)
    print("📝 Category Classifier Accuracy:", cat_clf.evaluate(test_df))

    # ---- Budget Recommender ----
    budget_rec = BudgetRecommender(model_path=os.path.join(MODEL_DIR, "budget_recommender.pkl"))
    budget_rec.train(df)
    recs = budget_rec.recommend(test_df)
    print("💡 Budget Recommendations Sample:", recs[:3])  # show first 3 recommendations

    print("\n✅ All models trained & saved to 'models/'")

if __name__ == "__main__":
    main()
