from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv

from src.rag_service import RAGService
from src.models import HackRXRequest, HackRXResponse

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Policy Reader RAG API",
    description="RAG-based API for policy document question answering",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Expected token
EXPECTED_TOKEN = "9d0a6a0d59a944b7b92b0a33b4cee5b30f2c00b4b098f133cfd1e36a90ada7d1"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials

# Initialize RAG service
rag_service = RAGService()

@app.get("/")
async def root():
    return {"message": "Policy Reader RAG API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/v1/hackrx/run", response_model=HackRXResponse)
async def run_hackrx(
    request: HackRXRequest,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """
    Process policy documents and answer questions using RAG
    """
    try:
        # Process the document and questions using RAG service
        answers = await rag_service.process_questions(request.documents, request.questions)
        
        return HackRXResponse(answers=answers)
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
