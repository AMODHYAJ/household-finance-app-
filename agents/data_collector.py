"""
Responsible AI & Commercialization (Template)
------------------------------------------------
* Fairness: Collects all user data as provided, no exclusion.
* Transparency: Logs all data collection activities.
* Explainability: Documents what data is collected and why.
* Data Protection: Encrypts and access-controls sensitive data.
* Commercialization: Premium connectors, automated data sync, business data feeds.
"""
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from core.database import SessionLocal, init_db, User, Transaction, Household
from core.security import hash_password, verify_password
from core.schemas import TransactionIn
from core.utils import parse_date
from models.category_classifier import CategoryClassifier

import pandas as pd
import json


class DataCollectorAgent:
    def __init__(self):
        init_db()
        self.db: Session = SessionLocal()
        self.current_user: User | None = None
        self.classifier = CategoryClassifier()

    # ---------- Auth ----------
    def register_user(self):
        username = input("Choose username: ").strip()
        password = input("Choose password: ").strip()
        role = input("Role (admin/user): ").strip().lower()
        if role not in ["admin", "user"]:
            role = "user"
        hh_choice = input("Create new household? (y/n): ").strip().lower()
        household = None
        if hh_choice == "y":
            hh_name = input("Household name: ").strip()
            household = Household(name=hh_name)
            self.db.add(household)
            self.db.flush()  # get id without commit
        else:
            existing = self.db.query(Household).all()
            if existing:
                print("Available households:")
                for h in existing:
                    print(f"  {h.id}: {h.name}")
                try:
                    hh_id = int(input("Enter household id to join (or 0 for none): ").strip() or "0")
                except ValueError:
                    hh_id = 0
                household = self.db.query(Household).get(hh_id) if hh_id else None

        user = User(username=username, password_hash=hash_password(password),
                    household=household, role=role)
        self.db.add(user)
        try:
            self.db.commit()
            print(f"✅ Registered successfully as {role}.")
        except IntegrityError:
            self.db.rollback()
            print("❌ Username already exists.")

    def delete_user_data(self):
        if not self.current_user or self.current_user.role != "admin":
            print("❌ Only admin users can delete data.")
            return
        confirm = input("Are you sure you want to delete all user data? (y/n): ").strip().lower()
        if confirm == "y":
            num_deleted = self.db.query(User).delete()
            self.db.commit()
            print(f"✅ Deleted {num_deleted} users and all related data.")
        else:
            print("Cancelled data deletion.")

    def login_user(self) -> bool:
        username = input("Username: ").strip()
        password = input("Password: ").strip()
        user = self.db.query(User).filter(User.username == username).first()
        if not user or not verify_password(password, user.password_hash):
            print("❌ Invalid credentials.")
            return False
        self.current_user = user
        print(f"✅ Welcome, {username}!")
        return True

    # ---------- Transactions ----------
    def add_transaction(self):
        if not self.current_user:
            print("Please login first.")
            return

        # Basic input
        t_type = input("Type (income/expense): ").strip().lower()
        if t_type not in ["income", "expense"]:
            print("❌ Invalid type. Must be 'income' or 'expense'.")
            return

        category = input("Category (leave blank for auto): ").strip()
        amount_s = input("Amount: ").strip()
        date_s = input("Date (YYYY-MM-DD): ").strip()
        note = input("Note (optional): ").strip()

        # Validation
        try:
            amount = float(amount_s)
            if amount <= 0:
                print("❌ Amount must be positive.")
                return
            date_val = parse_date(date_s)
        except Exception as e:
            print(f"❌ Invalid input: {e}")
            return

        # Auto-suggest category if blank but note is given
        if not category and note:
            try:
                category = self.classifier.predict(note)
                print(f"✨ Suggested category: {category}")
            except Exception:
                category = "uncategorized"

        # Duplicate detection (same date, type, amount, category, note)
        existing = (self.db.query(Transaction)
                    .filter(Transaction.user_id == self.current_user.id,
                            Transaction.t_type == t_type,
                            Transaction.amount == amount,
                            Transaction.date == date_val,
                            Transaction.category == category,
                            Transaction.note == (note or None))
                    .first())
        if existing:
            print("⚠️ Duplicate transaction detected. Skipping save.")
            return

        # Save
        tx = Transaction(
            user_id=self.current_user.id,
            t_type=t_type,
            category=category,
            amount=amount,
            date=date_val,
            note=note or None
        )
        self.db.add(tx)
        self.db.commit()
        print("✅ Transaction added.")

    def list_transactions(self):
        if not self.current_user:
            print("Please login first.")
            return []
        rows = (self.db.query(Transaction)
                .filter(Transaction.user_id == self.current_user.id)
                .order_by(Transaction.date.asc(), Transaction.id.asc())
                .all())
        if not rows:
            print("No transactions yet.")
        else:
            for r in rows:
                print(f"[{r.id}] {r.date} | {r.t_type} | {r.category} | {r.amount:.2f} | {r.note or ''}")
        return rows

    def get_transactions_df(self):
        rows = self.list_transactions()
        data = [{
            "id": r.id, "date": r.date, "type": r.t_type,
            "category": r.category, "amount": r.amount, "note": r.note
        } for r in rows]
        return pd.DataFrame(data)

    # ---------- Export ----------
    def export_to_csv(self, filename="transactions.csv"):
        df = self.get_transactions_df()
        if df.empty:
            print("No transactions to export.")
            return
        df.to_csv(filename, index=False)
        print(f"✅ Transactions exported to {filename}")

    def export_to_json(self, filename="transactions.json"):
        df = self.get_transactions_df()
        if df.empty:
            print("No transactions to export.")
            return
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
        print(f"✅ Transactions exported to {filename}")
