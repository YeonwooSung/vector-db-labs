# VectorDB and GraphDB - Hybrid Search System

PDF 문서를 처리하여 Vector DB, Elasticsearch, Graph DB에 저장하고 하이브리드 검색을 수행하는 시스템입니다.

## 🏗️ Architecture

```
PDF Documents
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│                     ETL Pipeline                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│  │ PDF Load │→ │ Chunking │→ │ Embedding│→ │ KG Extraction│ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
      │                │                            │
      ▼                ▼                            ▼
┌──────────┐    ┌──────────────┐             ┌──────────┐
│  Qdrant  │    │Elasticsearch │             │  Neo4j   │
│ (Vector) │    │   (BM25)     │             │ (Graph)  │
└──────────┘    └──────────────┘             └──────────┘
      │                │                            │
      └────────────────┼────────────────────────────┘
                       ▼
              ┌────────────────┐
              │ Hybrid Search  │
              │  (RRF/Weighted)│
              └────────────────┘
```

## 🚀 Quick Start

### 1. Docker 서비스 실행

```bash
# Qdrant + Elasticsearch 실행
docker-compose up -d qdrant es

# Neo4j도 함께 실행 (선택사항)
docker-compose --profile neo4j up -d
```

### 2. Python 환경 설정

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 3. 환경 변수 설정

```bash
# .env 파일 생성
cat > .env << EOF
# LLM API Keys (Knowledge Graph 추출용)
GOOGLE_API_KEY=your_google_api_key      # Gemini 사용시
ANTHROPIC_API_KEY=your_anthropic_key    # Claude 사용시

# Database connections (선택사항 - 기본값 사용 가능)
QDRANT_HOST=localhost
QDRANT_PORT=6333
ES_HOST=localhost
ES_PORT=9200
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
EOF
```

### 4. PDF 문서 준비

```bash
# docs 디렉토리에 PDF 파일 배치
cp /path/to/your/pdfs/*.pdf docs/
```

### 5. ETL 파이프라인 실행

```bash
# 기본 실행 (Gemini 사용)
python -m src.etl --docs-dir docs

# Anthropic Claude 사용
python -m src.etl --docs-dir docs --llm-provider anthropic

# Knowledge Graph 추출 없이 (LLM 불필요)
python -m src.etl --docs-dir docs --no-kg

# 기존 데이터 삭제 후 재처리
python -m src.etl --docs-dir docs --clear

# 청크 크기 조정
python -m src.etl --docs-dir docs --chunk-size 1024 --chunk-overlap 100

# 인덱스 타입 변경 (HNSW → Flat)
python -m src.etl --docs-dir docs --index-type flat
```

### 6. 검색 REPL 실행

```bash
# 검색 인터페이스 시작
python -m src.repl

# 결과 개수 조정
python -m src.repl --top-k 5

# 하이브리드 가중치 조정 (벡터 검색 비중)
python -m src.repl --alpha 0.7
```

## 📖 REPL 명령어

| 명령어 | 설명 |
|--------|------|
| `<query>` | 모든 DB에서 검색 후 하이브리드 결과 출력 |
| `/vector <query>` | Qdrant 벡터 검색만 |
| `/fts <query>` | Elasticsearch 전문 검색만 |
| `/graph <query>` | Neo4j 그래프 검색만 |
| `/hybrid <query>` | 벡터+FTS 하이브리드 검색만 |
| `/top <n>` | 결과 개수 설정 |
| `/stats` | DB 통계 조회 |
| `/help` | 도움말 |
| `/quit` | 종료 |

## 🔧 Configuration

### Qdrant 인덱스 설정

```python
from src.config import QdrantConfig, IndexType

# HNSW 인덱스 (기본값 - 빠른 검색)
config = QdrantConfig(
    index_type=IndexType.HNSW,
    hnsw_m=16,           # 노드당 엣지 수
    hnsw_ef_construct=100  # 구축 시 후보 리스트 크기
)

# Flat 인덱스 (정확한 검색, 작은 데이터셋에 적합)
config = QdrantConfig(index_type=IndexType.FLAT)
```

### 임베딩 설정

```python
from src.config import EmbeddingConfig

config = EmbeddingConfig(
    model_name="all-MiniLM-L6-v2",  # 기본 모델
    dimension=384
)
```

### 청킹 설정

```python
from src.config import ChunkingConfig

config = ChunkingConfig(
    chunk_size=512,      # 청크 크기 (문자)
    chunk_overlap=50,    # 오버랩 크기
    separator="\n"       # 분리자
)
```

## 📁 Project Structure

```
src/
├── __init__.py          # Package exports
├── config.py            # Configuration classes
├── etl.py              # ETL pipeline (main entry)
├── repl.py             # Search REPL interface
├── models/
│   └── __init__.py     # Data models (Document, Chunk, Entity, etc.)
├── loaders/
│   └── __init__.py     # PDF loader
├── processors/
│   └── __init__.py     # Text chunking & embedding
├── stores/
│   ├── __init__.py     # Base store interface
│   ├── qdrant_store.py # Qdrant vector store
│   ├── es_store.py     # Elasticsearch store
│   └── neo4j_store.py  # Neo4j graph store
├── extractors/
│   └── __init__.py     # Knowledge graph extraction (LLM)
└── search/
    └── __init__.py     # Hybrid search merger
```

## 🎯 Design Patterns Used

- **Repository Pattern**: 각 DB 접근을 추상화 (`BaseStore`)
- **Strategy Pattern**: 임베딩 모델, 인덱스 타입 교체 가능
- **Factory Pattern**: LLM 클라이언트 생성 (Gemini/Claude)
- **Facade Pattern**: ETL 및 검색 인터페이스 단순화

## 📊 Hybrid Search Algorithm

### Reciprocal Rank Fusion (RRF)

```
RRF_score = Σ (1 / (k + rank_i))
```

- `k`: 상수 (기본값 60)
- `rank_i`: 각 결과 리스트에서의 순위

### Weighted Scoring

```
Combined_score = α × vector_score + (1 - α) × fulltext_score
```

- `α`: 벡터 검색 가중치 (0-1, 기본값 0.5)

## 🐳 Docker Services

| Service | Port | Description |
|---------|------|-------------|
| Qdrant | 6333 | Vector database (HNSW index) |
| Elasticsearch | 9200 | Search engine (BM25) |
| Neo4j | 7474, 7687 | Graph database |
