"""
Parquet 데이터를 Qdrant로 로딩하는 스크립트
메모리 효율을 위해 청크 단위로 스트리밍 처리
"""

import gc
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from typing import Iterator
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.http import models

from schema import COLLECTION_NAME, create_collection


# 배치 사이즈 (Qdrant 업로드 단위)
BATCH_SIZE = 500

# 청크 사이즈 (Parquet 읽기 단위) - 메모리 사용량 조절
CHUNK_SIZE = 10000


def get_parquet_info(file_path: str) -> tuple:
    """Parquet 파일 정보 조회 (전체 로드 없이)"""
    parquet_file = pq.ParquetFile(file_path)
    total_rows = parquet_file.metadata.num_rows
    
    # 첫 번째 행만 읽어서 벡터 차원 확인
    first_batch = next(parquet_file.iter_batches(batch_size=1, columns=['vector']))
    sample_vector = first_batch.to_pandas()['vector'].iloc[0]
    
    if isinstance(sample_vector, np.ndarray):
        vector_dim = sample_vector.shape[0]
    elif isinstance(sample_vector, list):
        vector_dim = len(sample_vector)
    else:
        raise ValueError(f"알 수 없는 벡터 타입: {type(sample_vector)}")
    
    return total_rows, vector_dim


def iter_parquet_chunks(file_path: str, chunk_size: int = CHUNK_SIZE) -> Iterator[pd.DataFrame]:
    """Parquet 파일을 청크 단위로 스트리밍"""
    parquet_file = pq.ParquetFile(file_path)
    
    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        yield batch.to_pandas()
        gc.collect()  # 메모리 해제


def prepare_point(row: pd.Series, point_id: int) -> models.PointStruct:
    """DataFrame 행을 Qdrant PointStruct로 변환"""
    
    # 벡터 추출
    vector = row['vector']
    if isinstance(vector, np.ndarray):
        vector = vector.tolist()
    
    # node_info 처리
    node_info = None
    if 'node_info' in row and row['node_info'] is not None:
        node_info = {
            'start': row['node_info'][0] if isinstance(row['node_info'], (tuple, list)) else None,
            'end': row['node_info'][1] if isinstance(row['node_info'], (tuple, list)) and len(row['node_info']) > 1 else None,
        }
    
    # time 처리 (datetime -> Unix timestamp)
    time_value = row.get('time')
    if pd.notna(time_value):
        if hasattr(time_value, 'timestamp'):
            time_value = int(time_value.timestamp())
        elif isinstance(time_value, (int, float)):
            time_value = int(time_value)
        else:
            time_value = None
    else:
        time_value = None
    
    # 페이로드 구성
    payload = {
        'doc_id': int(row['doc_id']) if pd.notna(row.get('doc_id')) else None,
        'text': str(row['text']) if pd.notna(row.get('text')) else '',
        'node_info': node_info,
        'metadata': str(row['metadata']) if pd.notna(row.get('metadata')) else '',
        'type': str(row['type']) if pd.notna(row.get('type')) else 'comment',
        'by': str(row['by']) if pd.notna(row.get('by')) else '',
        'time': time_value,
        'title': str(row['title']) if pd.notna(row.get('title')) else '',
        'post_score': int(row['post_score']) if pd.notna(row.get('post_score')) else 0,
        'dead': bool(row['dead']) if pd.notna(row.get('dead')) else False,
        'deleted': bool(row['deleted']) if pd.notna(row.get('deleted')) else False,
        'length': int(row['length']) if pd.notna(row.get('length')) else 0,
    }
    
    # 원본 ID 사용 (있는 경우)
    original_id = row.get('id')
    if pd.notna(original_id):
        point_id = int(original_id)
    
    return models.PointStruct(
        id=point_id,
        vector=vector,
        payload=payload,
    )


def load_data_streaming(
    client: QdrantClient,
    file_path: str,
    batch_size: int = BATCH_SIZE,
    chunk_size: int = CHUNK_SIZE,
) -> int:
    """스트리밍 방식으로 데이터를 Qdrant에 로딩 (메모리 효율적)"""
    
    total_rows, _ = get_parquet_info(file_path)
    print(f"데이터 로딩 시작 (총 {total_rows:,} 행, 청크: {chunk_size}, 배치: {batch_size})")
    
    uploaded_count = 0
    global_idx = 0
    
    with tqdm(total=total_rows, desc="데이터 로딩") as pbar:
        for chunk_df in iter_parquet_chunks(file_path, chunk_size):
            batch = []
            
            for _, row in chunk_df.iterrows():
                point = prepare_point(row, global_idx)
                batch.append(point)
                global_idx += 1
                
                if len(batch) >= batch_size:
                    client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=batch,
                        wait=False,  # 비동기로 처리하여 속도 향상
                    )
                    pbar.update(len(batch))
                    uploaded_count += len(batch)
                    batch = []
            
            # 남은 배치 처리
            if batch:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=batch,
                    wait=False,
                )
                pbar.update(len(batch))
                uploaded_count += len(batch)
            
            # 청크 처리 후 메모리 해제
            del chunk_df
            gc.collect()
    
    print(f"총 {uploaded_count:,} 개의 포인트 로딩 완료!")
    return uploaded_count


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Parquet 데이터를 Qdrant로 로딩 (스트리밍)')
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/hackernews_part_1_of_1.parquet',
        help='Parquet 파일 경로'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Qdrant 호스트'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=6333,
        help='Qdrant 포트'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help='Qdrant 업로드 배치 크기'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=CHUNK_SIZE,
        help='Parquet 읽기 청크 크기 (메모리 조절)'
    )
    parser.add_argument(
        '--recreate',
        action='store_true',
        help='컬렉션 재생성 여부'
    )
    
    args = parser.parse_args()
    
    # Qdrant 클라이언트 연결
    print(f"Qdrant 연결: {args.host}:{args.port}")
    client = QdrantClient(host=args.host, port=args.port)
    
    # Parquet 파일 정보 조회 (메모리 로드 없이)
    print(f"Parquet 파일 분석: {args.data_path}")
    total_rows, vector_dim = get_parquet_info(args.data_path)
    print(f"총 행 수: {total_rows:,}")
    print(f"벡터 차원: {vector_dim}")
    
    # 컬렉션 재생성 옵션
    if args.recreate:
        try:
            client.delete_collection(collection_name=COLLECTION_NAME)
            print(f"기존 컬렉션 '{COLLECTION_NAME}' 삭제됨")
        except Exception:
            pass
    
    # 컬렉션 생성
    create_collection(client, vector_dimension=vector_dim)
    
    # 스트리밍 방식으로 데이터 로딩
    load_data_streaming(
        client=client,
        file_path=args.data_path,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )
    
    # 결과 확인
    collection_info = client.get_collection(collection_name=COLLECTION_NAME)
    print("\n=== 컬렉션 정보 ===")
    print(f"포인트 수: {collection_info.points_count:,}")
    print(f"상태: {collection_info.status}")


if __name__ == "__main__":
    main()
