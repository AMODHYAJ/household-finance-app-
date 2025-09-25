
import os
from cryptography.fernet import Fernet

# Use a fixed key for demo; in production, load from env/config
FERNET_KEY = os.environ.get("FERNET_KEY") or Fernet.generate_key()
fernet = Fernet(FERNET_KEY)

def encrypt_data(data: str) -> str:
    return fernet.encrypt(data.encode()).decode()

def decrypt_data(token: str) -> str:
    try:
        return fernet.decrypt(token.encode()).decode()
    except Exception:
        return token

def encrypt_amount(amount: float) -> str:
    return fernet.encrypt(str(amount).encode()).decode()

def decrypt_amount(token: str) -> float:
    if isinstance(token, float):
        return token
    try:
        return float(fernet.decrypt(token.encode()).decode())
    except Exception:
        return float(token)
import bcrypt

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False
