# Evaluation Metrics

## Overview
This document describes the evaluation metrics used to assess the performance, security, and Responsible AI compliance of the Household Finance AI Platform.

## 1. Functional Metrics
- **User Registration Success Rate:** Percentage of successful registrations vs. attempts.
- **Login Success Rate:** Percentage of successful logins vs. attempts.
- **Transaction Addition Success Rate:** Percentage of successful transaction entries vs. attempts.
- **Exportable Report Generation Rate:** Percentage of successful report exports vs. attempts.

## 2. Security Metrics
- **Input Validation Coverage:** Percentage of form fields with input sanitization and validation.
- **Password Hashing Coverage:** Percentage of user passwords stored as hashes (should be 100%).
- **Encryption Coverage:** Percentage of sensitive fields (notes, amounts) encrypted in the database.
- **Transparency Log Completeness:** Percentage of critical user actions recorded in the event log.
- **Account Deletion Success Rate:** Percentage of successful account deletions vs. attempts.

## 3. Responsible AI Metrics
- **Explainability Coverage:** Percentage of charts and insights with captions or explanations.
- **Privacy Control Usage:** Number of users utilizing data deletion/privacy features.
- **Role-Based Access Enforcement:** Percentage of admin/user actions correctly restricted by role.
- **Error Handling Coverage:** Percentage of user actions with robust error feedback.

## 4. Model Metrics (for ML features)
- **Expense Prediction Accuracy:** Measured by RMSE or MAE on test data.
- **Anomaly Detection Precision/Recall:** Evaluated on labeled test transactions.
- **Category Classification Accuracy:** Percentage of correct category predictions on test data.

## 5. Usability Metrics
- **User Satisfaction:** Collected via feedback forms or surveys.
- **Task Completion Time:** Average time to complete key actions (registration, transaction entry, report export).

## 6. Audit & Transparency Metrics
- **Event Log Retention:** Number of days/events retained in transparency logs.
- **Audit Trail Completeness:** Percentage of actions traceable via logs.

## How to Measure
- Use unit tests in `tests/` to validate functional and security metrics.
- Review transparency dashboard for log completeness.
- Analyze ML model performance using test datasets in `data/`.
- Collect user feedback for usability and satisfaction.

## Contact
For questions about evaluation or metrics, contact the project owner.
