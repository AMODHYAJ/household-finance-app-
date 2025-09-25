from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from core.database import SessionLocal, init_db, User, Transaction, Household
from core.security import hash_password, verify_password
from core.schemas import TransactionIn
from core.utils import parse_date

class DataCollectorAgent:
    def __init__(self):
        init_db()
        self.db: Session = SessionLocal()
        self.current_user: User | None = None

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
        t_type = input("Type (income/expense): ").strip().lower()
        category = input("Category (food, bills, etc.): ").strip()
        amount_s = input("Amount: ").strip()
        date_s = input("Date (YYYY-MM-DD): ").strip()
        note = input("Note (optional): ").strip()
        try:
            data = TransactionIn(
                t_type=t_type,
                category=category,
                amount=float(amount_s),
                date=parse_date(date_s),
                note=note or None
            )
        except Exception as e:
            print(f"❌ Invalid input: {e}")
            return

        from core.security import encrypt_data, encrypt_amount
        tx = Transaction(
            user_id=self.current_user.id,
            t_type=data.t_type,
            category=data.category,
            amount=encrypt_amount(data.amount),
            date=data.date,
            note=encrypt_data(data.note) if data.note else None
        )
        self.db.add(tx)
        self.db.commit()
        print("✅ Transaction added.")

    def list_transactions(self):
        if not self.current_user:
            print("Please login first.")
            return []
        from core.security import decrypt_data, decrypt_amount
        rows = (self.db.query(Transaction)
                .filter(Transaction.user_id == self.current_user.id)
                .order_by(Transaction.date.asc(), Transaction.id.asc())
                .all())
        if not rows:
            print("No transactions yet.")
        else:
            for r in rows:
                amount = decrypt_amount(r.amount)
                note = decrypt_data(r.note) if r.note else ''
                print(f"[{r.id}] {r.date} | {r.t_type} | {r.category} | {amount:.2f} | {note}")
        return rows

    def get_transactions_df(self):
        import pandas as pd
        rows = self.list_transactions()
        from core.security import decrypt_data, decrypt_amount
        data = [{
            "id": r.id, "date": r.date, "type": r.t_type,
            "category": r.category, "amount": decrypt_amount(r.amount), "note": decrypt_data(r.note) if r.note else ''
        } for r in rows]
        return pd.DataFrame(data)
