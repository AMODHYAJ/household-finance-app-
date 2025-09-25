# Security Documentation

## Overview
This document outlines the security features and practices implemented in the Household Finance AI Platform to protect user data, ensure privacy, and comply with Responsible AI guidelines.

## Authentication & Authorization
- **User Registration:** Input sanitization for username, password, and household name. Passwords must meet minimum length requirements.
- **Password Hashing:** All user passwords are hashed using bcrypt before storage. Plaintext passwords are never stored.
- **Role-Based Access Control:** Users are assigned roles (admin/user) with different privileges. Admins have access to management features; users have access to personal finance features.

## Data Protection
- **Encryption:** Sensitive transaction fields (notes, amounts) are encrypted using Fernet symmetric encryption. Only authorized users (and admins) with the key can decrypt and view sensitive data.
- **Input Validation:** All form inputs are sanitized to prevent injection attacks and ensure data integrity.
- **Account & Data Deletion:** Users can permanently delete their account and all associated data. This action is logged for transparency.

## Transparency & Auditability
- **Transparency Logs:** All critical user actions (registration, login, transaction addition, logout, account deletion) are recorded in an in-memory event log and displayed in the Transparency Dashboard for audit and accountability.
- **Error Handling:** All database and processing errors are caught and reported to the user. Failed transactions are rolled back to prevent data corruption.

## Privacy Controls
- **User Data Ownership:** Users have full control over their data, including the ability to delete their account and all transactions.
- **Limited Data Exposure:** Only authorized users and admins can view decrypted sensitive data. Other users cannot access encrypted fields.

## Secure Development Practices
- **Dependencies:** All dependencies are listed in `requirements.txt` and should be kept up to date to avoid vulnerabilities.
- **Code Quality:** Input validation, error handling, and logging are implemented throughout the codebase.

## Recommendations
- Use strong, unique passwords for each account.
- Regularly review transparency logs for suspicious activity.
- Keep the application and dependencies updated.

## Contact
For security concerns or questions, contact the project owner.
