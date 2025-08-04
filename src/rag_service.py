import os
import requests
import tempfile
import warnings
from typing import List, Dict, Any
import uuid
import time
import numpy as np
from groq import Groq
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Suppress FutureWarnings from transformers/torch
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*")

load_dotenv()

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
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "policy-reader")
        
        # Create index if it doesn't exist
        self._create_index_if_not_exists()
        
        # Initialize components
        # Using HuggingFace embeddings as free alternative to OpenAI
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def _create_index_if_not_exists(self):
        """Create Pinecone index if it doesn't exist"""
        try:
            # Get list of existing indexes using the new API
            existing_indexes = self.pc.list_indexes().names()
            
            if self.index_name not in existing_indexes:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=384,  # HuggingFace all-MiniLM-L6-v2 embedding dimension
                    metric="cosine",  # Best for text similarity (other options: euclidean, dotproduct)
                    spec=ServerlessSpec(
                        cloud="aws",  # Options: aws, gcp, azure
                        region="us-east-1"  # Choose region closest to your users
                    )
                    # Alternative: Pod-based spec for high-volume usage
                    # spec=PodSpec(
                    #     environment="us-east1-gcp",
                    #     pod_type="p1.x1"
                    # )
                )
                print(f"Created Pinecone index '{self.index_name}' with 384 dimensions and cosine metric")
                # Wait for index to be ready
                time.sleep(10)
            else:
                print(f"Pinecone index '{self.index_name}' already exists")
        except Exception as e:
            print(f"Warning: Could not create/check Pinecone index: {e}")
            # Continue without creating index - it might already exist
        
    def download_and_process_document(self, document_url: str) -> List:
        """Download PDF from URL and process it"""
        try:
            print(f"Downloading document from: {document_url}")
            # Download the PDF
            response = requests.get(document_url)
            response.raise_for_status()
            print(f"Downloaded {len(response.content)} bytes")
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name
            
            print(f"Saved to temporary file: {temp_path}")
            
            # Load and process the PDF
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            print(f"Loaded {len(documents)} pages from PDF")
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            # Split documents into chunks
            texts = self.text_splitter.split_documents(documents)
            print(f"Split into {len(texts)} text chunks")
            
            return texts
            
        except Exception as e:
            print(f"Error in download_and_process_document: {str(e)}")
            raise Exception(f"Error processing document: {str(e)}")
    
    def create_vector_store(self, texts: List[Document]) -> Dict[str, Any]:
        """Create vector store from document texts using Pinecone directly"""
        try:
            # Create unique namespace for this document
            namespace = f"doc_{str(uuid.uuid4())[:8]}"
            print(f"Creating vector store with namespace: {namespace}")
            
            # Get the Pinecone index
            index = self.pc.Index(self.index_name)
            print(f"Connected to Pinecone index: {self.index_name}")
            
            # Generate embeddings for all texts
            print(f"Generating embeddings for {len(texts)} documents...")
            text_contents = [doc.page_content for doc in texts]
            embeddings = self.embeddings.embed_documents(text_contents)
            
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
                        "values": embedding,
                        "metadata": {
                            "text": text.page_content,
                            "page": text.metadata.get("page", 0),
                            "source": text.metadata.get("source", "")
                        }
                    })
                
                # Upsert batch to Pinecone
                print(f"Upserting batch {batch_start+1}-{batch_end} of {total_vectors}")
                index.upsert(vectors=batch_vectors, namespace=namespace)
            
            print(f"Vector store created successfully with {total_vectors} vectors")
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
            query_embedding = vector_store["embeddings"].embed_query(query)
            
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
    
    async def process_questions(self, document_url: str, questions: List[str]) -> List[str]:
        """Process questions using Qwen for context+answering and Gemini for final output refinement."""
        try:
            print(f"Processing {len(questions)} questions")

            # Step 1: Document processing
            texts = self.download_and_process_document(document_url)
            vector_store = self.create_vector_store(texts)
            print(f"Vector store ready for querying")

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
