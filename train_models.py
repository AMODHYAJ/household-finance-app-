import pandas as pd
import os
from models.expense_predictor import ExpensePredictor
from models.anomaly_detector import AnomalyDetector
from models.category_classifier import CategoryClassifier

DATA_PATH = "data/finance_dataset.csv"
TEST_PATH = "data/finance_test.csv"
MODEL_DIR = "models/"

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("✅ Training with dataset:", len(df), "rows")

    exp_pred = ExpensePredictor(model_path=os.path.join(MODEL_DIR, "expense_predictor.pkl"))
    exp_pred.train(df)
    print("📈 Expense Predictor MAE:", exp_pred.evaluate(test_df))

    anomaly_det = AnomalyDetector(model_path=os.path.join(MODEL_DIR, "anomaly_detector.pkl"))
    anomaly_det.train(df)
    print("🚨 Anomalies in test set:", len(anomaly_det.detect(test_df)))

    cat_clf = CategoryClassifier(model_path=os.path.join(MODEL_DIR, "category_classifier.pkl"))
    cat_clf.train(df)
    print("📝 Category Classifier Accuracy:", cat_clf.evaluate(test_df))

    print("\n✅ All models trained & saved to 'models/'")

if __name__ == "__main__":
    main()
