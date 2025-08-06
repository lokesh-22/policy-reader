import os
import requests
import tempfile
import warnings
from typing import List, Dict, Any
import uuid
import time
import hashlib
import sqlite3
import json
import numpy as np
from groq import Groq
import PyPDF2
from io import BytesIO
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv

# Suppress FutureWarnings from transformers/torch
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*")

load_dotenv()

class Document:
    """Simple document class to replace LangChain Document"""
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

class SimpleTextSplitter:
    """Simple text splitter to replace LangChain's RecursiveCharacterTextSplitter"""
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks"""
        chunks = []
        for doc in documents:
            text_chunks = self._split_text(doc.page_content)
            for i, chunk in enumerate(text_chunks):
                chunks.append(Document(
                    page_content=chunk,
                    metadata={**doc.metadata, "chunk_id": i}
                ))
        return chunks
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        return chunks

class RAGService:
    def __init__(self):
        # Initialize Groq
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        # Initialize Pinecone
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        # Initialize Groq client
        self.groq_client = Groq(api_key=self.groq_api_key)
        self.model_name = "deepseek-r1-distill-llama-70b"  # Available models: mixtral-8x7b-32768, llama2-70b-4096
        
        # Initialize Pinecone using the new v3 API
        try:
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            self.index_name = os.getenv("PINECONE_INDEX_NAME", "policy-reader")
            print(f"Pinecone initialized with index name: '{self.index_name}'")
            
            # Create index if it doesn't exist
            self._create_index_if_not_exists()
        except Exception as e:
            print(f"Failed to initialize Pinecone: {e}")
            raise
        
        # Initialize components
        # Using SentenceTransformer directly instead of LangChain wrapper
        self.embeddings = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        self.text_splitter = SimpleTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Initialize SQLite database for caching
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for caching PDF processing results"""
        try:
            self.db_path = "pdf_cache.db"
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            
            # Create table for PDF processing cache
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS pdf_cache (
                    url_hash TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    vector_count INTEGER NOT NULL,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    pdf_hash TEXT,
                    metadata TEXT
                )
            """)
            
            # Create index for faster lookups
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_url_hash ON pdf_cache(url_hash)")
            self.conn.commit()
            print("SQLite database initialized for PDF caching")
            
            # Clean up old entries to keep database size manageable
            self._cleanup_old_cache_entries(50)  # Keep only 50 most recent entries
            
        except Exception as e:
            print(f"Warning: Could not initialize SQLite database: {e}")
            self.conn = None
    
    def _get_url_hash(self, url: str) -> str:
        """Generate hash for URL to use as cache key"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _get_cached_namespace(self, document_url: str) -> str:
        """Check if we have already processed this URL and return namespace"""
        if not self.conn:
            return None
            
        try:
            url_hash = self._get_url_hash(document_url)
            cursor = self.conn.execute(
                "SELECT namespace, vector_count FROM pdf_cache WHERE url_hash = ?",
                (url_hash,)
            )
            result = cursor.fetchone()
            
            if result:
                namespace, vector_count = result
                print(f"Found cached processing for URL: namespace={namespace}, vectors={vector_count}")
                
                # Verify the namespace still exists in Pinecone
                try:
                    index = self.pc.Index(self.index_name)
                    stats = index.describe_index_stats()
                    namespaces = stats.get('namespaces', {})
                    
                    if namespace in namespaces and namespaces[namespace]['vector_count'] > 0:
                        print(f"Verified namespace '{namespace}' exists in Pinecone with {namespaces[namespace]['vector_count']} vectors")
                        return namespace
                    else:
                        print(f"Namespace '{namespace}' not found in Pinecone, will reprocess")
                        # Remove stale cache entry
                        self.conn.execute("DELETE FROM pdf_cache WHERE url_hash = ?", (url_hash,))
                        self.conn.commit()
                        
                except Exception as e:
                    print(f"Error verifying namespace in Pinecone: {e}")
                    
            return None
            
        except Exception as e:
            print(f"Error checking cache: {e}")
            return None
    
    def _cache_processing_result(self, document_url: str, namespace: str, vector_count: int):
        """Cache the processing result for future use"""
        if not self.conn:
            return
            
        try:
            url_hash = self._get_url_hash(document_url)
            
            # Insert or replace cache entry
            self.conn.execute("""
                INSERT OR REPLACE INTO pdf_cache 
                (url_hash, url, namespace, vector_count, pdf_hash, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (url_hash, document_url, namespace, vector_count, url_hash, 
                  json.dumps({"processed_at": time.time()})))
            
            self.conn.commit()
            print(f"Cached processing result: URL={document_url[:50]}..., namespace={namespace}, vectors={vector_count}")
            
        except Exception as e:
            print(f"Error caching result: {e}")
    
    def _cleanup_old_cache_entries(self, max_entries: int = 100):
        """Clean up old cache entries to prevent database from growing too large"""
        if not self.conn:
            return
            
        try:
            # Keep only the most recent entries
            self.conn.execute("""
                DELETE FROM pdf_cache 
                WHERE url_hash NOT IN (
                    SELECT url_hash FROM pdf_cache 
                    ORDER BY processed_at DESC 
                    LIMIT ?
                )
            """, (max_entries,))
            
            deleted_count = self.conn.total_changes
            if deleted_count > 0:
                self.conn.commit()
                print(f"Cleaned up {deleted_count} old cache entries")
                
        except Exception as e:
            print(f"Error cleaning up cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache database"""
        if not self.conn:
            return {"status": "Database not available"}
            
        try:
            cursor = self.conn.execute("SELECT COUNT(*) FROM pdf_cache")
            total_entries = cursor.fetchone()[0]
            
            cursor = self.conn.execute("""
                SELECT COUNT(*) FROM pdf_cache 
                WHERE processed_at > datetime('now', '-24 hours')
            """)
            recent_entries = cursor.fetchone()[0]
            
            return {
                "total_cached_documents": total_entries,
                "processed_last_24h": recent_entries,
                "database_path": self.db_path
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _download_pdf_to_memory(self, document_url: str) -> bytes:
        """Download PDF directly to memory without using temp files"""
        try:
            print(f"Downloading PDF from: {document_url}")
            response = requests.get(document_url, timeout=30)
            response.raise_for_status()
            print(f"Downloaded {len(response.content)} bytes")
            return response.content
        except Exception as e:
            raise Exception(f"Error downloading PDF: {str(e)}")
    
    def _load_pdf_from_memory(self, pdf_content: bytes) -> List[Document]:
        """Load PDF from memory bytes using PyPDF2"""
        try:
            documents = []
            # Convert bytes to BytesIO for PyPDF2
            pdf_file = BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():  # Only add pages with content
                    documents.append(Document(
                        page_content=text,
                        metadata={"page": page_num + 1, "source": "pdf_download"}
                    ))
            return documents
        except Exception as e:
            raise Exception(f"Error loading PDF from memory: {str(e)}")
    
    def _create_index_if_not_exists(self):
        """Create Pinecone index if it doesn't exist using v3 API"""
        try:
            # Check if index exists using new API
            print(f"Checking for existing indexes...")
            active_indexes = [index.name for index in self.pc.list_indexes()]
            print(f"All active indexes: {active_indexes}")
            print(f"Looking for index named: '{self.index_name}'")
            
            if self.index_name not in active_indexes:
                print(f"Index '{self.index_name}' not found. Creating new index...")
                
                # Create index with new v3 API
                from pinecone import ServerlessSpec
                self.pc.create_index(
                    name=self.index_name,
                    dimension=384,  # for all-MiniLM-L6-v2
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                
                print(f"Index '{self.index_name}' created successfully")
                # Wait for index to be ready
                print("Waiting 15 seconds for index to be ready...")
                import time
                time.sleep(15)
                
                # Check if index is now ready
                updated_indexes = [index.name for index in self.pc.list_indexes()]
                print(f"Updated index list: {updated_indexes}")
            else:
                print(f"Index '{self.index_name}' already exists. Skipping creation.")
                
            # Additional check: try to get index details
            try:
                index_info = self.pc.describe_index(self.index_name)
                print(f"Index '{self.index_name}' details: {index_info}")
            except Exception as e:
                print(f"Could not get index details: {e}")
                
        except Exception as e:
            print(f"Warning: Could not create/check Pinecone index: {e}")
            # Continue without creating index - it might already exist
        
    def download_and_process_document(self, document_url: str) -> tuple[List[Document], str]:
        """Download PDF from URL and process it with caching support"""
        try:
            # Check if we have already processed this URL
            cached_namespace = self._get_cached_namespace(document_url)
            if cached_namespace:
                print(f"Using cached processing result for this URL")
                # Return empty list for texts since we'll use cached vectors
                return [], cached_namespace
            
            print(f"Processing new document from: {document_url}")
            
            # Download PDF to memory (no temp files)
            pdf_content = self._download_pdf_to_memory(document_url)
            
            # Load and process the PDF from memory
            documents = self._load_pdf_from_memory(pdf_content)
            print(f"Loaded {len(documents)} pages from PDF")
            
            # Split documents into chunks
            texts = self.text_splitter.split_documents(documents)
            print(f"Split into {len(texts)} text chunks")
            
            return texts, None  # None indicates this is new processing
            
        except Exception as e:
            print(f"Error in download_and_process_document: {str(e)}")
            raise Exception(f"Error processing document: {str(e)}")
    
    def _load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF using PyPDF2 from file path"""
        documents = []
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                documents.append(Document(
                    page_content=text,
                    metadata={"page": page_num + 1, "source": file_path}
                ))
        return documents
    
    def create_vector_store(self, texts: List[Document], cached_namespace: str = None, document_url: str = None) -> Dict[str, Any]:
        """Create vector store from document texts using Pinecone with caching support"""
        try:
            # If we have a cached namespace, use it
            if cached_namespace:
                print(f"Using cached vector store with namespace: {cached_namespace}")
                index = self.pc.Index(self.index_name)
                return {
                    "index": index,
                    "namespace": cached_namespace,
                    "embeddings": self.embeddings
                }
            
            # Create unique namespace for this document
            namespace = f"doc_{str(uuid.uuid4())[:8]}"
            print(f"Creating new vector store with namespace: {namespace}")
            
            # Get the Pinecone index using new v3 API
            try:
                print(f"Attempting to connect to index: '{self.index_name}'")
                index = self.pc.Index(self.index_name)
                print(f"Successfully connected to Pinecone index: '{self.index_name}'")
                
                # Test the connection by getting index stats
                print("Testing connection with index stats...")
                stats = index.describe_index_stats()
                print(f"Index stats - Total vectors: {stats.get('total_vector_count', 'Unknown')}")
                print(f"Index dimension: {stats.get('dimension', 'Unknown')}")
                
                # Check if index is ready
                if 'total_vector_count' not in stats:
                    print("Warning: Index may not be fully ready yet")
                    
            except Exception as e:
                print(f"Failed to connect to index '{self.index_name}': {e}")
                print(f"Error type: {type(e).__name__}")
                
                # List available indexes for debugging
                try:
                    available_indexes = [index.name for index in self.pc.list_indexes()]
                    print(f"Available indexes in your account: {available_indexes}")
                    
                    if available_indexes:
                        print("Suggestion: Check if your index name matches one of the available indexes above")
                    else:
                        print("No indexes found in your account. You may need to create one first.")
                        
                except Exception as list_error:
                    print(f"Could not list available indexes: {list_error}")
                raise
            
            # Generate embeddings for all texts
            print(f"Generating embeddings for {len(texts)} documents...")
            text_contents = [doc.page_content for doc in texts]
            embeddings = self.embeddings.encode(text_contents)
            
            # Prepare vectors for upsert in batches
            batch_size = 100  # Safe batch size for 384-dim vectors
            total_vectors = len(texts)
            
            print(f"Upserting {total_vectors} vectors in batches of {batch_size}...")
            
            for batch_start in range(0, total_vectors, batch_size):
                batch_end = min(batch_start + batch_size, total_vectors)
                batch_texts = texts[batch_start:batch_end]
                batch_embeddings = embeddings[batch_start:batch_end]
                
                # Prepare batch vectors
                batch_vectors = []
                for i, (text, embedding) in enumerate(zip(batch_texts, batch_embeddings)):
                    batch_vectors.append({
                        "id": f"{namespace}_{batch_start + i}",
                        "values": embedding.tolist(),  # Convert numpy array to list
                        "metadata": {
                            "text": text.page_content,
                            "page": text.metadata.get("page", 0),
                            "source": text.metadata.get("source", "")
                        }
                    })
                
                # Upsert batch to Pinecone with retry logic
                print(f"Upserting batch {batch_start+1}-{batch_end} of {total_vectors}")
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        index.upsert(vectors=batch_vectors, namespace=namespace)
                        break  # Success, exit retry loop
                    except Exception as e:
                        if attempt == max_retries - 1:  # Last attempt
                            raise e
                        print(f"Attempt {attempt + 1} failed, retrying in 5 seconds: {e}")
                        time.sleep(5)
            
            print(f"Vector store created successfully with {total_vectors} vectors")
            
            # Cache the result if we have a document URL
            if document_url:
                self._cache_processing_result(document_url, namespace, total_vectors)
            
            return {
                "index": index,
                "namespace": namespace,
                "embeddings": self.embeddings
            }
                
        except Exception as e:
            print(f"Error in create_vector_store: {str(e)}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Error creating vector store: {str(e)}")
    
    def retrieve_documents(self, query: str, vector_store: Dict[str, Any], k: int = 6) -> List[Document]:
        """Retrieve relevant documents from vector store"""
        try:
            # Generate embedding for query
            query_embedding = vector_store["embeddings"].encode([query])[0].tolist()  # Convert to list
            
            # Query Pinecone
            results = vector_store["index"].query(
                vector=query_embedding,
                top_k=k,
                namespace=vector_store["namespace"],
                include_metadata=True
            )
            
            # Convert results to Document objects
            documents = []
            for match in results.matches:
                doc = Document(
                    page_content=match.metadata["text"],
                    metadata={
                        "page": match.metadata.get("page", 0),
                        "source": match.metadata.get("source", ""),
                        "score": match.score
                    }
                )
                documents.append(doc)
            
            return documents
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            return []
    
    def _retrieve_from_pinecone(self, query_embedding, vector_store: Dict[str, Any], k: int) -> List[Document]:
        """Retrieve documents from Pinecone"""
        try:
            # Query Pinecone
            results = vector_store["index"].query(
                vector=query_embedding,
                top_k=k,
                namespace=vector_store["namespace"],
                include_metadata=True
            )
            
            # Convert results to Document objects
            documents = []
            for match in results.matches:
                doc = Document(
                    page_content=match.metadata["text"],
                    metadata={
                        "page": match.metadata.get("page", 0),
                        "source": match.metadata.get("source", ""),
                        "score": match.score
                    }
                )
                documents.append(doc)
            
            return documents
        except Exception as e:
            print(f"Error retrieving from Pinecone: {e}")
            return []
    
    async def process_questions(self, document_url: str, questions: List[str]) -> List[str]:
        """Process questions using Qwen for context+answering and Gemini for final output refinement."""
        try:
            print(f"Processing {len(questions)} questions")

            # Step 1: Document processing with caching
            texts, cached_namespace = self.download_and_process_document(document_url)
            
            if cached_namespace:
                # Use cached vectors
                vector_store = self.create_vector_store([], cached_namespace, document_url)
                print(f"Using cached vector store for querying")
            else:
                # Process new document
                vector_store = self.create_vector_store(texts, None, document_url)
                print(f"Created new vector store for querying")

            final_answers = []

            for i, question in enumerate(questions):
                try:
                    print(f"\n---\nQuestion {i+1}: {question[:60]}...")

                    # Step 2: Retrieve relevant context
                    relevant_docs = self.retrieve_documents(question, vector_store, k=6)
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])
                    print(f"Retrieved {len(relevant_docs)} relevant documents")

                    # Step 3: Prompt Qwen with few-shot prompt
                    qwen_prompt = f"""
    You are a professional assistant helping extract direct policy answers from insurance documents. 
    Always respond in 1–2 clear, complete sentences using policy language. Avoid any internal thoughts or reasoning.

    Examples:

    Context:
    A grace period of thirty days is allowed for payment of renewal premium without losing continuity.

    Question:
    What is the grace period for premium payment under the policy?

    Answer:
    A grace period of thirty days is provided for premium payment after the due date.

    ---

    Context:
    Pre-existing diseases are covered after 36 months of continuous policy coverage.

    Question:
    What is the waiting period for pre-existing diseases?

    Answer:
    There is a waiting period of thirty-six (36) months for pre-existing diseases from the first policy inception.

    ---

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

                    chat_completion = self.groq_client.chat.completions.create(
                        messages=[{"role": "user", "content": qwen_prompt}],
                        model=self.model_name,
                        temperature=0,
                        max_tokens=100,
                    )
                    qwen_answer = chat_completion.choices[0].message.content.strip()
                    print(f"Qwen Answer: {qwen_answer}")

                    # Step 4: Refine using Gemini
                    gemini_prompt = f"""
    You are a professional assistant refining insurance policy answers.

    Here is a question about a policy, some relevant context from the policy document, and a draft answer.

    Context:
    {context}

    Question:
    {question}

    Draft Answer:
    {qwen_answer}

    Instructions:
    - Evaluate the draft answer.
    - Rewrite it as a single, complete, policy-style sentence.
    - Be specific with numbers and timeframes.
    - Do not include reasoning steps or thoughts.
    - Keep it under 50 words.

    Final Answer:
    """

                    final_answer = self.gemini_generate_answer(gemini_prompt)
                    print(f"Gemini Final Answer: {final_answer}")

                    final_answers.append(final_answer)

                except Exception as e:
                    error_msg = f"Error processing question {i+1}: {str(e)}"
                    print(error_msg)
                    final_answers.append(error_msg)

            return final_answers

        except Exception as e:
            raise Exception(f"Error in RAG pipeline: {str(e)}")
    
    def gemini_generate_answer(self, prompt: str) -> str:
        """Use Gemini to refine or validate the answer from Qwen."""
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not set")
        genai.configure(api_key=gemini_api_key)

        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
