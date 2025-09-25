# Household Finance AI Platform

## Overview
This application is a multi-agent, privacy-focused household finance management system built with Streamlit, SQLAlchemy, and FastAPI. It supports secure user registration, login, transaction management, explainable charts, exportable reports, and transparency logs for Responsible AI compliance.

## Key Features
- **User Registration & Login:** Secure, with input sanitization and password hashing.
- **Transaction Management:** Add, view, and categorize income/expense transactions. Input validation and encryption for sensitive fields.
- **Role-Based Access:** Admin and user roles with different privileges.
- **Charts & Insights:** Interactive visualizations and AI-generated insights.
- **Exportable Reports:** Downloadable summaries and charts.
- **Transparency Dashboard:** Logs user actions (login, registration, transaction addition, logout, deletion) for accountability.
- **Privacy Controls:** Users can delete their own account and data.

## Transparency Logs
All major user actions are recorded in an in-memory event log (`user_event_log`), including:
- Registration
- Login
- Transaction addition
- Logout
- Account deletion

These logs are displayed in the Transparency Dashboard for audit and accountability.

## Getting Started
1. Install dependencies from `requirements.txt`.
2. Run `streamlit run streamlit_app.py` to launch the app.
3. Register a new user, log in, and start managing transactions.

## Security & Responsible AI
- Input sanitization for all forms
- Passwords are hashed
- Sensitive transaction fields are encrypted
- Transparency logs for all critical actions

## File Structure
- `streamlit_app.py`: Main UI and logic
- `core/`: Database models, security, utilities
- `agents/`: Modular agent classes
- `models/`: ML models for insights
- `visualizations/`: Static charts
- `tests/`: Unit tests

## Contact
For questions or support, contact the project owner.
