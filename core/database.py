from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey, Enum
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from config.settings import DATABASE_URL
from datetime import date

Base = declarative_base()

class Household(Base):
    __tablename__ = "households"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    users = relationship("User", back_populates="household")

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    household_id = Column(Integer, ForeignKey("households.id"), nullable=True)
    household = relationship("Household", back_populates="users")
    transactions = relationship("Transaction", back_populates="user")

class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    t_type = Column(String, nullable=False)  # "income" or "expense"
    category = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    date = Column(Date, nullable=False, default=date.today)
    note = Column(String, nullable=True)
    user = relationship("User", back_populates="transactions")

engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def init_db():
    Base.metadata.create_all(engine)
