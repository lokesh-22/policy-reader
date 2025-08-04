from pydantic import BaseModel
from typing import List

class HackRXRequest(BaseModel):
    documents: str  # URL to the policy document
    questions: List[str]

class HackRXResponse(BaseModel):
    answers: List[str]
