from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
from agents.insight_generator import InsightGeneratorAgent
import pandas as pd
import spacy
from jose import JWTError, jwt
import re
import os


app = FastAPI()

# JWT config
SECRET_KEY = os.environ.get("API_SECRET_KEY", "supersecretkey")
ALGORITHM = "HS256"

security = HTTPBearer()

# Request model for transaction data

class Transaction(BaseModel):
	date: str
	type: str
	amount: float
	category: str
	note: str | None = None

	@validator('date')
	def date_format(cls, v):
		# Simple YYYY-MM-DD check
		if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
			raise ValueError("date must be in YYYY-MM-DD format")
		return v

	@validator('type')
	def type_valid(cls, v):
		if v not in {"income", "expense"}:
			raise ValueError("type must be 'income' or 'expense'")
		return v

	@validator('amount')
	def amount_positive(cls, v):
		if v < 0:
			raise ValueError("amount must be positive")
		return v

	@validator('category')
	def category_not_empty(cls, v):
		if not v or not v.strip():
			raise ValueError("category must not be empty")
		return v


class TransactionsRequest(BaseModel):
	transactions: list[Transaction]


insight_agent = InsightGeneratorAgent()

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")


def set_security_headers(response: Response):
	response.headers["X-Content-Type-Options"] = "nosniff"
	response.headers["X-Frame-Options"] = "DENY"
	response.headers["X-XSS-Protection"] = "1; mode=block"
	response.headers["Content-Security-Policy"] = "default-src 'self'"

def verify_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)):
	token = credentials.credentials
	try:
		payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
		return payload
	except JWTError:
		raise HTTPException(status_code=401, detail="Invalid or expired token.")

@app.post("/summarize-transactions")
def summarize_transactions(request: TransactionsRequest, response: Response, user=Depends(verify_jwt)):
	set_security_headers(response)
	# Concatenate notes and categories for NLP summarization
	texts = [f"{t.category}: {t.note}" for t in request.transactions if t.note]
	if not texts:
		return {"summary": "No transaction notes to summarize."}
	doc = nlp(". ".join(texts))
	# Extract entities and keywords
	entities = [(ent.text, ent.label_) for ent in doc.ents]
	keywords = list(set([token.lemma_ for token in doc if token.is_alpha and not token.is_stop]))
	summary = {
		"entities": entities,
		"keywords": keywords,
		"text": doc.text
	}
	return {"summary": summary}


@app.post("/generate-insights")
def generate_insights(request: TransactionsRequest, response: Response, user=Depends(verify_jwt)):
	set_security_headers(response)
	# Convert transactions to DataFrame
	df = pd.DataFrame([t.dict() for t in request.transactions])
	insights = insight_agent.generate_insights(df)
	return {"insights": insights}

# Example login endpoint to get JWT (for demo only)
from fastapi import status
class LoginRequest(BaseModel):
	username: str
	password: str

@app.post("/login")
def login(request: LoginRequest, response: Response):
	set_security_headers(response)
	# TODO: Replace with real user lookup and password check
	if request.username == "demo" and request.password == "demo":
		token = jwt.encode({"sub": request.username}, SECRET_KEY, algorithm=ALGORITHM)
		return {"access_token": token}
	raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
