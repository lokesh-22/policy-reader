#!/usr/bin/env python3
"""
Quick test to verify pinecone-client version compatibility
"""

try:
    print("Testing pinecone-client import...")
    import pinecone
    from pinecone import Pinecone
    print("✅ Basic imports successful")
    
    try:
        from pinecone import ServerlessSpec
        print("✅ ServerlessSpec available (newer API)")
    except ImportError:
        print("⚠️  ServerlessSpec not available (older API - will use fallback)")
    
    print("✅ Pinecone imports working correctly")
    
except ImportError as e:
    print(f"❌ Pinecone import failed: {e}")

try:
    print("\nTesting other critical imports...")
    import fastapi
    import uvicorn
    import sentence_transformers
    import transformers
    import torch
    import groq
    import google.generativeai
    print("✅ All critical imports successful")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")

print("\n🎉 Requirements test completed!")
