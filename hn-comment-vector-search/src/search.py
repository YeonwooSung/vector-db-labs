"""
Qdrant 벡터 검색 REPL 스크립트
SentenceTransformer를 사용한 쿼리 임베딩 및 검색
"""

import argparse
from typing import List, Optional
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

from schema import COLLECTION_NAME


# 임베딩 모델
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# 대규모 데이터셋 검색 설정
SEARCH_TIMEOUT = 60            # 검색 타임아웃 (초)
HNSW_EF_SEARCH = 128           # 검색 시 탐색 범위 (높을수록 정확, 느림)


def load_embedding_model() -> SentenceTransformer:
    """SentenceTransformer 모델 로드"""
    print(f"임베딩 모델 로딩: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"모델 로드 완료 (벡터 차원: {model.get_sentence_embedding_dimension()})")
    return model


def encode_query(model: SentenceTransformer, query: str) -> List[float]:
    """쿼리를 벡터로 인코딩"""
    embedding = model.encode(query, convert_to_numpy=True)
    return embedding.tolist()


def search_by_vector(
    client: QdrantClient,
    query_vector: List[float],
    limit: int = 10,
    score_threshold: Optional[float] = None,
    filter_type: Optional[str] = None,
    filter_by: Optional[str] = None,
    min_score: Optional[int] = None,
) -> List[models.ScoredPoint]:
    """
    벡터로 유사한 댓글 검색
    
    Args:
        client: Qdrant 클라이언트
        query_vector: 쿼리 벡터
        limit: 반환할 결과 수
        score_threshold: 최소 유사도 점수
        filter_type: 타입 필터 (story, comment, poll, pollopt, job)
        filter_by: 작성자 필터
        min_score: 최소 post_score 필터
    """
    
    # 필터 조건 구성
    must_conditions = []
    
    if filter_type:
        must_conditions.append(
            models.FieldCondition(
                key="type",
                match=models.MatchValue(value=filter_type),
            )
        )
    
    if filter_by:
        must_conditions.append(
            models.FieldCondition(
                key="by",
                match=models.MatchValue(value=filter_by),
            )
        )
    
    if min_score is not None:
        must_conditions.append(
            models.FieldCondition(
                key="post_score",
                range=models.Range(gte=min_score),
            )
        )

    query_filter = None
    if must_conditions:
        query_filter = models.Filter(must=must_conditions)

    # 검색 실행 (대규모 데이터셋 최적화)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        score_threshold=score_threshold,
        query_filter=query_filter,
        with_payload=True,
        search_params=models.SearchParams(
            hnsw_ef=HNSW_EF_SEARCH,  # 검색 정확도 조절
            exact=False,             # 근사 검색 사용 (빠름)
        ),
        timeout=SEARCH_TIMEOUT,
    )
    
    return results.points


def search_by_id(
    client: QdrantClient,
    point_id: int,
    limit: int = 10,
) -> List[models.ScoredPoint]:
    """
    특정 포인트와 유사한 댓글 검색 (ID 기반)
    """
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=models.RecommendQuery(positive=[point_id]),
        limit=limit,
        with_payload=True,
    )
    
    return results.points


def format_result(point: models.ScoredPoint, rank: int) -> str:
    """검색 결과 포맷팅"""
    payload = point.payload
    
    # 시간 포맷팅
    time_str = ""
    if payload.get('time'):
        time_str = datetime.fromtimestamp(payload['time']).strftime('%Y-%m-%d %H:%M')
    
    text = payload.get('text', '')
    if len(text) > 300:
        text = text[:300] + "..."
    
    return f"""
[{rank}] Score: {point.score:.4f} | ID: {point.id}
    Type: {payload.get('type', 'N/A')} | By: {payload.get('by', 'N/A')} | Time: {time_str}
    Post Score: {payload.get('post_score', 0)} | Length: {payload.get('length', 0)}
    Title: {payload.get('title', '')[:80] if payload.get('title') else '-'}
    ────────────────────────────────────────
    {text}
"""


def search_repl(client: QdrantClient, model: SentenceTransformer, limit: int = 10):
    """REPL 방식 검색 인터페이스"""
    
    print("\n" + "=" * 60)
    print("  HackerNews Comment Vector Search")
    print("=" * 60)
    
    # 컬렉션 정보 확인
    try:
        info = client.get_collection(collection_name=COLLECTION_NAME)
        print(f"  컬렉션: {COLLECTION_NAME}")
        print(f"  총 포인트 수: {info.points_count:,}")
        print(f"  벡터 차원: {info.config.params.vectors.size}")
    except Exception as e:
        print(f"  컬렉션 연결 실패: {e}")
        return
    
    print("=" * 60)
    print("  명령어:")
    print("    - 검색어 입력: 유사한 댓글 검색")
    print("    - /limit N: 결과 수 변경 (현재: {})".format(limit))
    print("    - /type TYPE: 타입 필터 (story/comment/poll/job)")
    print("    - /clear: 필터 초기화")
    print("    - /quit 또는 /exit: 종료")
    print("=" * 60 + "\n")
    
    current_filter_type = None
    current_limit = limit
    
    while True:
        try:
            query = input("🔍 검색> ").strip()
            
            if not query:
                continue
            
            # 명령어 처리
            if query.startswith('/'):
                parts = query.split()
                cmd = parts[0].lower()
                
                if cmd in ('/quit', '/exit', '/q'):
                    print("검색을 종료합니다.")
                    break
                
                elif cmd == '/limit' and len(parts) > 1:
                    try:
                        current_limit = int(parts[1])
                        print(f"결과 수: {current_limit}")
                    except ValueError:
                        print("올바른 숫자를 입력하세요.")
                    continue
                
                elif cmd == '/type' and len(parts) > 1:
                    current_filter_type = parts[1]
                    print(f"타입 필터: {current_filter_type}")
                    continue
                
                elif cmd == '/clear':
                    current_filter_type = None
                    print("필터가 초기화되었습니다.")
                    continue
                
                else:
                    print("알 수 없는 명령어입니다.")
                    continue
            
            # 쿼리 임베딩
            print("임베딩 생성 중...")
            query_vector = encode_query(model, query)
            
            # 검색 실행
            results = search_by_vector(
                client=client,
                query_vector=query_vector,
                limit=current_limit,
                filter_type=current_filter_type,
            )
            
            if not results:
                print("검색 결과가 없습니다.\n")
                continue
            
            # 결과 출력 (유사도 순으로 정렬되어 있음)
            print(f"\n{'─' * 60}")
            print(f"검색 결과: {len(results)}개 (유사도 순)")
            print(f"{'─' * 60}")

            for rank, result in enumerate(results, 1):
                print(format_result(result, rank))

            print()

        except KeyboardInterrupt:
            print("\n검색을 종료합니다.")
            break
        except Exception as e:
            print(f"오류 발생: {e}\n")


def main():
    parser = argparse.ArgumentParser(description='HackerNews Comment Vector Search REPL')
    parser.add_argument('--host', type=str, default='localhost', help='Qdrant 호스트')
    parser.add_argument('--port', type=int, default=6333, help='Qdrant 포트')
    parser.add_argument('--limit', type=int, default=10, help='기본 검색 결과 수')

    args = parser.parse_args()

    # Qdrant 클라이언트 연결
    print(f"Qdrant 연결: {args.host}:{args.port}")
    client = QdrantClient(host=args.host, port=args.port, timeout=60)

    # 임베딩 모델 로드
    model = load_embedding_model()

    # REPL 시작
    search_repl(client, model, limit=args.limit)


if __name__ == "__main__":
    main()
