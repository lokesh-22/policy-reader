#!/usr/bin/env python3
"""
Startup script for the Policy Reader RAG API
"""
import uvicorn
import os
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "True").lower() == "true"
    
    print(f"Starting Policy Reader RAG API on {host}:{port}")
    print("Make sure to set your GROQ_API_KEY in the .env file")
    
    # Start the server
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        access_log=True
    )

if __name__ == "__main__":
    main()
