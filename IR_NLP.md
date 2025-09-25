# Advanced IR/NLP Features

## Overview
The Household Finance AI Platform includes advanced Information Retrieval (IR) and Natural Language Processing (NLP) features to provide explainable, actionable financial insights.

## NLP Features
- **Category Classification:**
  - Uses a Naive Bayes classifier (with TF-IDF vectorization) to automatically classify transaction notes into categories (e.g., food, bills, salary).
  - Trained on historical transaction data; predicts category for new notes.
  - Accuracy is evaluated and reported during model training.

- **Budget Recommendations:**
  - Analyzes spending patterns and provides personalized recommendations (e.g., reduce top category spending, save a percentage of income).

## IR Features
- **Anomaly Detection:**
  - Uses Isolation Forest to detect unusual expenses in transaction data.
  - Flags anomalies and provides explainability for detected outliers.

- **Expense Prediction:**
  - Uses linear regression to forecast next month's expenses based on historical data.
  - Provides explainable predictions for budgeting.

## Explainability
- All insights and predictions include explainability captions, describing the logic or model used.
- Example: "This classification uses a Naive Bayes model trained on transaction notes and categories."

## How It Works
- Models are trained via `train_models.py` using CSV datasets.
- Predictions, classifications, and recommendations are generated in real time via the `InsightGeneratorAgent`.
- Results are displayed in the dashboard and insights pages, with explanations for transparency.

## Extensibility
- The system can be extended with more advanced NLP (e.g., spaCy, transformers) for richer note analysis, entity extraction, or sentiment analysis.
- IR can be expanded to include semantic search, document retrieval, or personalized query answering.

## References
- See `agents/insight_generator.py`, `models/category_classifier.py`, `models/anomaly_detector.py`, `models/expense_predictor.py`, and `models/budget_recommender.py` for implementation details.

## Contact
For questions or to extend IR/NLP features, contact the project owner.
