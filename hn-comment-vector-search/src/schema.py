"""
Qdrant 스키마 및 컬렉션 설정
HackerNews 댓글 벡터 검색을 위한 스키마 정의
"""

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PayloadSchemaType,
)


# 컬렉션 이름
COLLECTION_NAME = "hackernews_comments"

# 벡터 차원 (all-MiniLM-L6-v2 모델 기준)
VECTOR_DIMENSION = 384

# 대규모 데이터셋 설정 (~28M 포인트)
HNSW_M = 16                    # 그래프 연결 수 (메모리 vs 정확도 트레이드오프)
HNSW_EF_CONSTRUCT = 200        # 인덱스 구축 시 탐색 범위 (높을수록 정확, 느림)
INDEXING_THRESHOLD = 50000     # 인덱싱 임계값


def create_collection(client: QdrantClient, vector_dimension: int = VECTOR_DIMENSION) -> None:
    """
    HackerNews 댓글을 위한 Qdrant 컬렉션 생성
    
    ClickHouse 스키마 매핑:
    - id -> point id
    - doc_id -> payload.doc_id
    - text -> payload.text
    - vector -> vector
    - node_info -> payload.node_info
    - metadata -> payload.metadata
    - type -> payload.type (story, comment, poll, pollopt, job)
    - by -> payload.by (작성자)
    - time -> payload.time (Unix timestamp)
    - title -> payload.title
    - post_score -> payload.post_score
    - dead -> payload.dead
    - deleted -> payload.deleted
    - length -> payload.length
    """
    
    # 기존 컬렉션 확인 및 삭제 (선택적)
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if COLLECTION_NAME in collection_names:
        print(f"컬렉션 '{COLLECTION_NAME}'이 이미 존재합니다.")
        return

    # 컬렉션 생성
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_dimension,
            distance=Distance.COSINE,  # 코사인 유사도 사용
        ),
        # HNSW 인덱스 설정 (대규모 데이터셋 최적화)
        hnsw_config=models.HnswConfigDiff(
            m=HNSW_M,
            ef_construct=HNSW_EF_CONSTRUCT,
        ),
        # 최적화 설정
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=INDEXING_THRESHOLD,
        ),
    )

    print(f"컬렉션 '{COLLECTION_NAME}' 생성 완료!")
    
    # 페이로드 인덱스 생성 (필터링 성능 향상)
    _create_payload_indexes(client)


def _create_payload_indexes(client: QdrantClient) -> None:
    """필터링을 위한 페이로드 인덱스 생성"""

    # type 필드 인덱스 (story, comment 등으로 필터링)
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="type",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    # by 필드 인덱스 (작성자로 필터링)
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="by",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    # time 필드 인덱스 (시간 범위 필터링)
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="time",
        field_schema=PayloadSchemaType.INTEGER,
    )

    # post_score 필드 인덱스 (점수로 필터링)
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="post_score",
        field_schema=PayloadSchemaType.INTEGER,
    )

    # doc_id 필드 인덱스
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="doc_id",
        field_schema=PayloadSchemaType.INTEGER,
    )
    
    print("페이로드 인덱스 생성 완료!")


def delete_collection(client: QdrantClient) -> None:
    """컬렉션 삭제"""
    client.delete_collection(collection_name=COLLECTION_NAME)
    print(f"컬렉션 '{COLLECTION_NAME}' 삭제 완료!")


def get_collection_info(client: QdrantClient) -> dict:
    """컬렉션 정보 조회"""
    info = client.get_collection(collection_name=COLLECTION_NAME)
    return {
        "name": COLLECTION_NAME,
        "vectors_count": info.vectors_count,
        "points_count": info.points_count,
        "status": info.status,
        "vector_size": info.config.params.vectors.size,
        "distance": info.config.params.vectors.distance,
    }


if __name__ == "__main__":
    # 테스트용 실행
    client = QdrantClient(host="localhost", port=6333)
    create_collection(client)
    print(get_collection_info(client))
