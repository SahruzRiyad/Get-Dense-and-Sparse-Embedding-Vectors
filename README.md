# Text Embedding Service

A local service to generate dense and sparse (BM25) text embeddings using FastEmbed and rank_bm25.

---

## Base URL
```
http://localhost:8000
```

---

## Authentication
All protected endpoints require an API key in the request header:

```
X-API-Key: your_api_key_here
```

---

## Endpoints

### 1. Root Endpoint
**GET** `/` – Returns basic service information.

**Request:**
- Method: `GET`
- Authentication: Not required

**Response:**
```json
{
  "service": "Text Embedding Service",
  "status": "running",
  "version": "1.0.1",
  "timestamp": 1693276800.123
}
```

---

### 2. Health Check

**GET** `/health` – Returns service health status including model availability.

**Response:**

```json
{
  "status": "healthy",
  "models_loaded": true,
  "dense_available": true,
  "sparse_fastembed_available": true,
  "sparse_bm25_available": true,
  "timestamp": 1693276800.123
}
```

**Response Fields:**

* `status`: `"healthy"` or `"degraded"`
* `models_loaded`: Boolean indicating if FastEmbed models are loaded
* `dense_available`: Boolean indicating dense model availability
* `sparse_fastembed_available`: Boolean indicating FastEmbed sparse model availability
* `sparse_bm25_available`: Boolean indicating BM25 (rank_bm25) availability
* `timestamp`: Unix timestamp

---

### 3. Service Information

**GET** `/info` – Returns service configuration and model info.

**Request:**

* Method: `GET`
* Authentication: **Required**
* Header:

  ```
  X-API-Key: your_api_key_here
  ```

**Response:**

```json
{
  "dense_model": "BAAI/bge-small-en",
  "sparse_model": "Qdrant/bm25",
  "fastembed_available": true,
  "rank_bm25_available": true,
  "max_text_length": 8192,
  "max_batch_size": 32,
  "models_loaded": true
}
```

---

### 4. Generate Embeddings

**POST** `/embed` – Generates dense or sparse embeddings with model selection.

**Request:**

* Method: `POST`
* Authentication: **Required**
* Headers:

  ```
  Content-Type: application/json
  X-API-Key: your_api_key_here
  ```

**Request Body:**

```json
{
  "texts": ["Hello world", "Another example text"],
  "embedding_type": "dense"
}
```

**For Sparse Embeddings:**

```json
{
  "texts": ["Hello world", "Another example text"],
  "embedding_type": "sparse",
  "sparse_model": "fastembed"
}
```

**Request Parameters:**

* `texts`: Array of strings to embed (1-32 items, max 8192 chars each)
* `embedding_type`: `"dense"` or `"sparse"`
* `sparse_model`: `"fastembed"` or `"bm25"` (optional, defaults to "fastembed" when embedding_type is "sparse")

**Response (Dense):**

```json
{
  "vectors": [[0.1234, -0.5678, 0.9012, ...], [0.7890, 0.2468, -0.1357, ...]],
  "model": "BAAI/bge-small-en",
  "embedding_type": "dense",
  "processing_time_ms": 245.67
}
```

**Response (Sparse):**

```json
{
  "vectors": [
    {"indices": [0, 15, 23], "values": [2.5, 1.8, 3.2]},
    {"indices": [5, 12, 28], "values": [1.9, 2.7, 1.4]}
  ],
  "model": "Qdrant/bm25",
  "embedding_type": "sparse",
  "processing_time_ms": 123.45
}
```

---

### 5. Validate Input

**POST** `/validate` – Validates texts without generating embeddings.

**Request Body:**

```json
{
  "texts": ["Sample text", "Another text"],
  "embedding_type": "sparse",
  "sparse_model": "bm25"
}
```

**Response:**

```json
{
  "valid": true,
  "text_count": 2,
  "embedding_type": "sparse",
  "sparse_model": "bm25",
  "timestamp": 1693276800.123
}
```

---

## Error Responses

### Authentication Errors

* **Missing API Key (401)**

```json
{"detail": "API Key is missing. Please provide X-API-Key header.", "code": 401}
```

* **Invalid API Key (401)**

```json
{"detail": "Invalid API Key.", "code": 401}
```

### Validation Errors

* Invalid embedding type (400)

```json
{"detail": "Invalid embedding_type. Use 'dense' or 'sparse'.", "code": 400}
```

* Invalid sparse model (400)

```json
{"detail": "Invalid sparse_model. Use 'fastembed' or 'bm25'.", "code": 400}
```

* Text too long (422)
* Empty texts array (422)
* Too many texts (422)

### Service Errors

* Model not available (503)
* Embedding generation error (500)
* Unexpected error (500)

---

## Example Usage

### cURL

```bash
# Get Service Status
curl -X GET "http://localhost:8000/"

# Health Check
curl -X GET "http://localhost:8000/health"

# Get Service Info
curl -X GET "http://localhost:8000/info" -H "X-API-Key: your_api_key_here"

# Generate Dense Embeddings
curl -X POST "http://localhost:8000/embed" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{"texts":["Hello world","How are you?"],"embedding_type":"dense"}'

# Generate Sparse Embeddings (FastEmbed)
curl -X POST "http://localhost:8000/embed" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{"texts":["Machine learning","NLP"],"embedding_type":"sparse","sparse_model":"fastembed"}'

# Generate Sparse Embeddings (BM25)
curl -X POST "http://localhost:8000/embed" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{"texts":["Machine learning","NLP"],"embedding_type":"sparse","sparse_model":"bm25"}'
```

### Python

```python
import requests

BASE_URL = "http://localhost:8000"
API_KEY = "your_api_key_here"
HEADERS = {"Content-Type": "application/json", "X-API-Key": API_KEY}

# Dense embeddings
response = requests.post(
    f"{BASE_URL}/embed",
    headers=HEADERS,
    json={"texts": ["Hello world", "How are you?"], "embedding_type": "dense"}
)

# Sparse embeddings with FastEmbed
response = requests.post(
    f"{BASE_URL}/embed",
    headers=HEADERS,
    json={
        "texts": ["Machine learning", "Natural language processing"], 
        "embedding_type": "sparse",
        "sparse_model": "fastembed"
    }
)

# Sparse embeddings with BM25
response = requests.post(
    f"{BASE_URL}/embed",
    headers=HEADERS,
    json={
        "texts": ["Machine learning", "Natural language processing"], 
        "embedding_type": "sparse",
        "sparse_model": "bm25"
    }
)

if response.ok:
    result = response.json()
    print(f"Generated {len(result['vectors'])} embeddings using {result['model']}")
    print(f"Processing time: {result['processing_time_ms']:.2f}ms")
else:
    print(response.json())
```

### JavaScript

```javascript
const BASE_URL = "http://localhost:8000";
const API_KEY = "your_api_key_here";

async function generateEmbeddings(texts, type, sparseModel = null) {
  const body = {texts, embedding_type: type};
  if (type === "sparse" && sparseModel) {
    body.sparse_model = sparseModel;
  }
  
  const res = await fetch(`${BASE_URL}/embed`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json', 'X-API-Key': API_KEY},
    body: JSON.stringify(body)
  });
  
  if (!res.ok) throw await res.json();
  return await res.json();
}

// Dense embeddings
generateEmbeddings(["Hello world", "How are you?"], "dense")
  .then(console.log)
  .catch(console.error);

// Sparse embeddings with FastEmbed
generateEmbeddings(["Machine learning", "NLP"], "sparse", "fastembed")
  .then(console.log)
  .catch(console.error);

// Sparse embeddings with BM25
generateEmbeddings(["Machine learning", "NLP"], "sparse", "bm25")
  .then(console.log)
  .catch(console.error);
```

---

## Sparse Model Options

When using `embedding_type: "sparse"`, you can choose between two sparse models:

### FastEmbed Sparse (`sparse_model: "fastembed"`)
- Uses the FastEmbed library's sparse embedding model
- Requires FastEmbed to be installed
- Generally more accurate and consistent
- Default option if `sparse_model` is not specified

### BM25 (`sparse_model: "bm25"`)
- Uses the rank_bm25 library
- Creates BM25 model dynamically from input texts
- Good fallback option when FastEmbed is not available
- Requires rank_bm25 to be installed

---

## Configuration

Environment variables for customizing the service:

```env
API_KEY=your_secure_api_key_here
DENSE_MODEL_NAME=BAAI/bge-small-en
SPARSE_MODEL_NAME=Qdrant/bm25
MAX_TEXT_LENGTH=8192
MAX_BATCH_SIZE=32
REQUEST_TIMEOUT=30.0
```

---

## Installation & Dependencies

```bash
# Install FastEmbed (recommended)
pip install fastembed

# Install rank_bm25 for BM25 sparse embeddings
pip install rank-bm25

# Install other dependencies
pip install fastapi uvicorn python-dotenv
```

---

## Running the Service

```bash
# Start the service
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# The service will be available at http://localhost:8000
# API documentation at http://localhost:8000/docs
```