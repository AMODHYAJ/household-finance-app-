from pydantic import BaseModel, Field, validator
from datetime import date

class TransactionIn(BaseModel):
    t_type: str = Field(..., pattern="^(income|expense)$")
    category: str
    amount: float = Field(..., gt=0)
    date: date
    note: str | None = None

    @validator("category")
    def category_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Category cannot be empty")
        return v.strip()
