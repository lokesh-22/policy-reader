# Deployment Guide for Render

## 🚀 Render Deployment with SQLite Caching

This application is optimized for cloud deployment on Render with the following features:

### ✅ **Cloud-Ready Features**
- **No temporary files**: PDFs processed directly in memory
- **SQLite caching**: Persistent storage for processed documents
- **Pinecone integration**: Scalable vector database
- **Automatic cleanup**: Prevents database bloat

### 📁 **File Structure for Deployment**
```
policyReader/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── src/
│   ├── rag_service.py   # Main RAG service with caching
│   └── models.py        # Pydantic models
├── .env                 # Environment variables
└── pdf_cache.db         # SQLite database (auto-created)
```

### 🔧 **Environment Variables for Render**

Set these in your Render dashboard:

```bash
# Required API Keys
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key

# Configuration
PINECONE_INDEX_NAME=policy-readerr
API_TOKEN=9d0a6a0d59a944b7b92b0a33b4cee5b30f2c00b4b098f133cfd1e36a90ada7d1

# Server Settings (Render defaults)
HOST=0.0.0.0
PORT=10000
RELOAD=False
```

### 🎯 **Deploy Steps**

1. **Create Render Web Service**
   - Connect your GitHub repository
   - Choose "Web Service"
   - Set build command: `pip install -r requirements.txt`
   - Set start command: `python main.py`

2. **Configure Environment Variables**
   - Add all the environment variables listed above
   - Make sure API keys are correct

3. **Deploy**
   - Render will automatically build and deploy
   - First startup may take 2-3 minutes (downloading ML models)

### 📊 **Caching System**

The SQLite database provides:
- **Fast responses**: Cached documents load instantly
- **Cost savings**: No need to reprocess same PDFs
- **Persistent storage**: Cache survives deployments
- **Automatic cleanup**: Keeps only 50 most recent documents

### 🔍 **Monitor Cache**

Check cache status:
```bash
GET /cache/stats
Authorization: Bearer 9d0a6a0d59a944b7b92b0a33b4cee5b30f2c00b4b098f133cfd1e36a90ada7d1
```

Response:
```json
{
  "success": true,
  "cache_stats": {
    "total_cached_documents": 15,
    "processed_last_24h": 8,
    "database_path": "pdf_cache.db"
  }
}
```

### 🛠 **Performance Optimizations**

1. **Memory Usage**: PDFs processed in memory (no temp files)
2. **Vector Reuse**: Cached documents skip embedding generation
3. **Batch Processing**: Efficient Pinecone upserts
4. **Retry Logic**: Robust error handling for network issues

### 📈 **Scaling Considerations**

- **SQLite limit**: Handles thousands of documents efficiently
- **Memory**: Each PDF uses ~10-50MB during processing
- **Pinecone**: Unlimited vector storage in the cloud
- **Response time**: Cached documents: <1s, New documents: 10-30s

### 🔄 **How Caching Works**

1. **URL Hashing**: Each PDF URL gets a unique hash
2. **Namespace Mapping**: Hash maps to Pinecone namespace
3. **Cache Check**: Before processing, check if URL was processed
4. **Reuse Vectors**: If cached, use existing Pinecone vectors
5. **Auto Cleanup**: Old entries removed to prevent bloat

### 🚨 **Troubleshooting**

**Slow first request?**
- First deployment downloads ML models (~2-3 minutes)
- Subsequent requests are fast

**Out of memory?**
- Large PDFs (>100MB) may cause issues
- Consider reducing batch size in code

**Cache not working?**
- Check `/cache/stats` endpoint
- Verify SQLite database is writable

**Pinecone errors?**
- Verify API key and index name
- Check if index exists in Pinecone console

### 💡 **Best Practices**

1. **Monitor cache stats** regularly
2. **Use same PDF URLs** to benefit from caching
3. **Set appropriate timeout** for large PDFs
4. **Monitor Pinecone usage** to control costs
