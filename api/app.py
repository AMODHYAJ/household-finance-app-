from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agents.insight_generator import InsightGeneratorAgent
import pandas as pd
import spacy

app = FastAPI()

# Request model for transaction data
class Transaction(BaseModel):
	date: str
	type: str
	amount: float
	category: str
	note: str | None = None

class TransactionsRequest(BaseModel):
	transactions: list[Transaction]

insight_agent = InsightGeneratorAgent()

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

@app.post("/summarize-transactions")
def summarize_transactions(request: TransactionsRequest):
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
def generate_insights(request: TransactionsRequest):
	# Convert transactions to DataFrame
	df = pd.DataFrame([t.dict() for t in request.transactions])
	insights = insight_agent.generate_insights(df)
	return {"insights": insights}
