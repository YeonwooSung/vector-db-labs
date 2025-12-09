# HN Comments Vector Search

A vector search system utilizing Hacker News comment embedding data.

Download data from [here](https://clickhouse-datasets.s3.amazonaws.com/hackernews-miniLM/hackernews_part_1_of_1.parquet)

## 🗄️ Database Selection

**Why Qdrant was chosen:**
- Purpose-built database optimized for vector search
- Fast Approximate Nearest Neighbor search with HNSW index
- Supports metadata-based filtering combined with vector search
- Easy deployment with Docker

## 🚀 Getting Started

### 1. Start Qdrant Server

```bash
docker-compose up -d
```

Qdrant will be running at `http://localhost:6333`.

### 2. Set Up Python Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Load Data

```bash
cd src
python load_data.py --data-path ../../data/hackernews_part_1_of_1.parquet --recreate
```

### 4. Test Search

```bash
python search.py --demo
```

## 📊 Schema Information

### Qdrant Collection: `hackernews_comments`

| Field | Type | Description |
|-------|------|-------------|
| `id` | Point ID | Original HN item ID |
| `vector` | Float32[] | Embedding vector |
| `doc_id` | Integer | Document ID |
| `text` | String | Comment/story text |
| `type` | Keyword | story, comment, poll, pollopt, job |
| `by` | Keyword | Author |
| `time` | Integer | Unix timestamp |
| `title` | String | Title (for stories) |
| `post_score` | Integer | Score |
| `dead` | Boolean | Dead status |
| `deleted` | Boolean | Deleted status |
| `length` | Integer | Text length |

### Indexes

- **Vector Index**: HNSW (Cosine similarity)
- **Payload Indexes**: `type`, `by`, `time`, `post_score`, `doc_id`

## 🔍 Search Example

```python
from qdrant_client import QdrantClient
from src.search import search_by_vector

client = QdrantClient(host="localhost", port=6333)

# Search for similar comments by vector
results = search_by_vector(
    client=client,
    query_vector=your_embedding,
    limit=10,
    filter_type="comment",  # Filter comments only
    min_score=10,           # Minimum score of 10
)
```

## 🐳 Docker Commands

```bash
# Start
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f qdrant

# Stop
docker-compose down

# Remove including data
docker-compose down -v
```

## 📝 API Endpoints

Qdrant REST API: `http://localhost:6333`
- Dashboard: `http://localhost:6333/dashboard`
- Collection Info: `GET /collections/hackernews_comments`
- Search: `POST /collections/hackernews_comments/points/search`

## References

- [Hacker News vector search dataset](https://clickhouse.com/docs/getting-started/example-datasets/hackernews-vector-search-dataset)
- [Hacker News 댓글 2,800만 건을 벡터 임베딩 검색 데이터세트로 제공](https://news.hada.io/topic?id=24703&utm_source=slack&utm_medium=bot&utm_campaign=T01QNFF90J1)
