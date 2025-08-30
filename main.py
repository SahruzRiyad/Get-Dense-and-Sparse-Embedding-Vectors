import os
import re
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Union
import time

from fastapi import FastAPI, Header, HTTPException, status, Depends
from pydantic import BaseModel, Field, field_validator
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import hashlib
import hmac
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import FastEmbed for sparse (BM25) and dense embeddings
try:
    from fastembed import TextEmbedding, SparseTextEmbedding
    FASTEMBED_AVAILABLE = True
    logger.info("FastEmbed is available")
except ImportError:
    logger.warning("FastEmbed not installed. Falling back to rank_bm25 for sparse embeddings.")
    FASTEMBED_AVAILABLE = False
    TextEmbedding = None
    SparseTextEmbedding = None

# Import rank_bm25 for sparse embeddings option
try:
    from rank_bm25 import BM25Okapi
    RANK_BM25_AVAILABLE = True
    logger.info("rank_bm25 is available")
except ImportError:
    logger.error("rank_bm25 is not installed. BM25 sparse embedding functionality will be disabled.")
    RANK_BM25_AVAILABLE = False
    BM25Okapi = None

# ======================================
# Configuration
# ======================================
class Settings(BaseModel):
    api_key: str = Field("01977@Rashed_Embeddings", env="API_KEY", description="API Key for authentication")
    dense_model_name: str = Field("BAAI/bge-small-en", env="DENSE_MODEL_NAME", description="Name of the dense embedding model")
    sparse_model_name: str = Field("Qdrant/bm25", env="SPARSE_MODEL_NAME", description="Name of the sparse embedding model (FastEmbed only)")
    sparse_model_name_2: str = Field("rank_bm25", env="SPARSE_MODEL_NAME_2", description="Name of the sparse embedding model (Rank_BM25)" )
    max_text_length: int = Field(8192, env="MAX_TEXT_LENGTH", description="Maximum length of input text")
    max_batch_size: int = Field(32, env="MAX_BATCH_SIZE", description="Maximum number of texts per request")
    request_timeout: float = Field(30.0, env="REQUEST_TIMEOUT", description="Request timeout in seconds")
    
    class Config:
        env_file = ".env"

settings = Settings()

# Log the generated API key if it's auto-generated (for development)
if not os.getenv("API_KEY"):
    logger.warning(f"Using auto-generated API key: {settings.api_key}")

# ======================================
# FastAPI App Lifecycle Management
# ======================================
class GlobalModels:
    """Class to hold global model instances."""
    dense_model: Optional[TextEmbedding] = None
    sparse_fastembed_model: Optional[SparseTextEmbedding] = None
    models_loaded: bool = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    Initializes models when the app starts.
    """
    logger.info("FastAPI app starting up...")
    
    if FASTEMBED_AVAILABLE:
        try:
            logger.info(f"Loading dense model: {settings.dense_model_name}")
            GlobalModels.dense_model = TextEmbedding(model_name=settings.dense_model_name)
            logger.info("Dense model loaded successfully.")

            logger.info(f"Loading sparse FastEmbed model: {settings.sparse_model_name}")
            GlobalModels.sparse_fastembed_model = SparseTextEmbedding(model_name=settings.sparse_model_name)
            logger.info("Sparse FastEmbed model loaded successfully.")
            
            GlobalModels.models_loaded = True
        except Exception as e:
            logger.error(f"Error loading FastEmbed models: {e}")
            # Continue startup - other models may still work
    else:
        logger.info("FastEmbed not available. Dense and FastEmbed sparse models will not be loaded.")

    yield
    
    logger.info("FastAPI app shutting down...")
    # Clean up resources
    GlobalModels.dense_model = None
    GlobalModels.sparse_fastembed_model = None
    GlobalModels.models_loaded = False
    logger.info("Resources cleaned up successfully")

# ======================================
# FastAPI App Instance
# ======================================
app = FastAPI(
    title="Text Embedding Service",
    version="1.0.1",
    description="A production-ready service for generating dense and sparse (BM25) text embeddings.",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================
# Pydantic Models for Request/Response
# ======================================
class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=settings.max_batch_size, 
                            example=["hello world", "how are you?"])
    embedding_type: str = Field(..., example="dense", 
                               description="Type of embedding: 'dense' or 'sparse'.")
    sparse_model: Optional[str] = Field("fastembed", example="fastembed",
                                      description="Sparse model to use: 'fastembed' or 'bm25'. Only used when embedding_type is 'sparse'.")
    
    @field_validator('texts')
    @classmethod
    def validate_text_length(cls, v):
        for text in v:
            if len(text) > settings.max_text_length:
                raise ValueError(f"Text length exceeds maximum of {settings.max_text_length} characters")
            if not text.strip():
                raise ValueError("Empty or whitespace-only text is not allowed")
        return v
    
    @field_validator('embedding_type')
    @classmethod
    def validate_embedding_type(cls, v):
        if v not in ['dense', 'sparse']:
            raise ValueError("embedding_type must be 'dense' or 'sparse'")
        return v
    
    @field_validator('sparse_model')
    @classmethod
    def validate_sparse_model(cls, v):
        if v is not None and v not in ['fastembed', 'bm25']:
            raise ValueError("sparse_model must be 'fastembed' or 'bm25'")
        return v

class SparseVector(BaseModel):
    indices: List[int]
    values: List[float]

class EmbedResponse(BaseModel):
    vectors: List[Union[List[float], SparseVector]]
    model: str
    embedding_type: str
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    dense_available: bool
    sparse_fastembed_available: bool
    sparse_bm25_available: bool
    timestamp: float

class InfoResponse(BaseModel):
    dense_model: str
    sparse_model: str
    fastembed_available: bool
    rank_bm25_available: bool
    max_text_length: int
    max_batch_size: int
    models_loaded: bool

# ======================================
# Authentication and Security
# ======================================
def constant_time_compare(a: str, b: str) -> bool:
    """Constant-time string comparison to prevent timing attacks."""
    return hmac.compare_digest(a.encode(), b.encode())

def check_auth(x_api_key: Optional[str] = Header(None)):
    """Authenticates requests using an API key with constant-time comparison."""
    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key is missing. Please provide X-API-Key header."
        )
    if not constant_time_compare(x_api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key."
        )

# ======================================
# Custom Exception Handlers
# ======================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom handler for HTTPExceptions to standardize error responses."""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "code": exc.status_code, "timestamp": time.time()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Custom handler for unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected error occurred.", 
            "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "timestamp": time.time()
        }
    )

# ======================================
# Helper Functions
# ======================================
def generate_dense_embeddings(texts: List[str]) -> tuple[List[List[float]], str]:
    """Generate dense embeddings using FastEmbed."""
    if not GlobalModels.dense_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dense embedding model is not loaded."
        )
    
    try:
        vectors = [vec.tolist() for vec in GlobalModels.dense_model.embed(texts)]
        return vectors, settings.dense_model_name
    except Exception as e:
        logger.error(f"Error during dense embedding generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during dense embedding generation: {str(e)}"
        )

def generate_sparse_fastembed_embeddings(texts: List[str]) -> tuple[List[SparseVector], str]:
    """Generate sparse embeddings using FastEmbed."""
    if not GlobalModels.sparse_fastembed_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="FastEmbed sparse embedding model is not loaded."
        )
    
    try:
        vectors_raw = GlobalModels.sparse_fastembed_model.embed(texts)
        vectors = [
            SparseVector(
                indices=vec.indices.tolist(), 
                values=vec.values.tolist()
            )
            for vec in vectors_raw
        ]
        return vectors, settings.sparse_model_name
    except Exception as e:
        logger.error(f"Error during FastEmbed sparse embedding generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during FastEmbed sparse embedding generation: {str(e)}"
        )

def simple_tokenize(text: str) -> List[str]:
    """
    Tokenizer: lowercase + alphanumeric words only.
    Prevents punctuation tokens that would cause empty vectors.
    """
    return re.findall(r"\b\w+\b", text.lower())

def generate_sparse_bm25_embeddings(texts: List[str]) -> tuple[List[SparseVector], str]:
    """Generate sparse embeddings using BM25 (per-document pseudo-vectors)."""
    if not RANK_BM25_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="BM25 (rank_bm25) is not available."
        )

    try:
        # Tokenize
        tokenized_texts = [simple_tokenize(t) for t in texts]

        # Init BM25 with corpus
        bm25 = BM25Okapi(tokenized_texts)

        # Vocabulary index
        vocab = sorted(bm25.idf.keys())
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}

        vectors = []
        for doc_tokens in tokenized_texts:
            indices, values = [], []
            if not doc_tokens:
                vectors.append(SparseVector(indices=[], values=[]))
                continue

            # Score each term in this document as if it's the query
            for term in set(doc_tokens):
                score = bm25.get_scores([term])[tokenized_texts.index(doc_tokens)]
                if score > 1e-8:
                    indices.append(word_to_idx[term])
                    values.append(float(score))

            vectors.append(SparseVector(indices=indices, values=values))

        return vectors, "rank_bm25"

    except Exception as e:
        logger.error(f"Error during BM25 sparse embedding generation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during BM25 sparse embedding generation: {str(e)}"
        )





# ======================================
# Endpoints
# ======================================
@app.post("/embed", response_model=EmbedResponse, summary="Generate text embeddings")
def embed(request: EmbedRequest, _: str = Depends(check_auth)):
    """
    Generates dense or sparse text embeddings for a list of texts.

    - **texts**: A list of strings to embed (max length per text: {max_text_length} chars).
    - **embedding_type**: Specify "dense" for dense embeddings or "sparse" for sparse embeddings.
    - **sparse_model**: When embedding_type is "sparse", specify "fastembed" or "bm25" for the sparse model to use.
    
    Returns embeddings with processing metadata.
    """
    start_time = time.time()
    
    try:
        if request.embedding_type == "dense":
            vectors, model_name = generate_dense_embeddings(request.texts)
            
        elif request.embedding_type == "sparse":
            # Default to fastembed if sparse_model not specified
            sparse_model = request.sparse_model or "fastembed"
            
            if sparse_model == "fastembed":
                vectors, model_name = generate_sparse_fastembed_embeddings(request.texts)
            elif sparse_model == "bm25":
                vectors, model_name = generate_sparse_bm25_embeddings(request.texts)
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid sparse_model. Use 'fastembed' or 'bm25'."
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid embedding_type. Use 'dense' or 'sparse'."
            )
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return EmbedResponse(
            vectors=vectors,
            model=model_name,
            embedding_type=request.embedding_type,
            processing_time_ms=processing_time_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in embed endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during embedding generation."
        )

@app.get("/health", response_model=HealthResponse, summary="Health check endpoint")
def health():
    """
    Returns the service health status including model availability.
    """
    dense_available = GlobalModels.dense_model is not None
    sparse_fastembed_available = GlobalModels.sparse_fastembed_model is not None
    sparse_bm25_available = RANK_BM25_AVAILABLE
    
    return HealthResponse(
        status="healthy" if (dense_available or sparse_fastembed_available or sparse_bm25_available) else "degraded",
        models_loaded=GlobalModels.models_loaded,
        dense_available=dense_available,
        sparse_fastembed_available=sparse_fastembed_available,
        sparse_bm25_available=sparse_bm25_available,
        timestamp=time.time()
    )

@app.get("/", summary="Root endpoint")
def root():
    """
    Returns basic service information.
    """
    return {
        "service": "Text Embedding Service",
        "status": "running",
        "version": "1.0.1",
        "timestamp": time.time()
    }

@app.get("/info", response_model=InfoResponse, summary="Get service information")
def get_info(_: str = Depends(check_auth)):
    """
    Returns detailed service configuration and model information.
    Requires authentication.
    """
    return InfoResponse(
        dense_model=settings.dense_model_name,
        sparse_model=f"{settings.sparse_model_name} | {settings.sparse_model_name_2}",
        fastembed_available=FASTEMBED_AVAILABLE,
        rank_bm25_available=RANK_BM25_AVAILABLE,
        max_text_length=settings.max_text_length,
        max_batch_size=settings.max_batch_size,
        models_loaded=GlobalModels.models_loaded
    )

# ======================================
# Additional Utility Endpoints
# ======================================
@app.post("/validate", summary="Validate input without generating embeddings")
def validate_input(request: EmbedRequest, _: str = Depends(check_auth)):
    """
    Validates input texts without generating embeddings.
    Useful for checking if texts meet length and format requirements.
    """
    return {
        "valid": True,
        "text_count": len(request.texts),
        "embedding_type": request.embedding_type,
        "sparse_model": request.sparse_model if request.embedding_type == "sparse" else None,
        "timestamp": time.time()
    }