"""
Workit - 법령 KB Qdrant 마이그레이션 스크립트
로컬의 vectors.npz와 chunks.json을 읽어 Qdrant law_kb 컬렉션에 Hybrid 형태로 적재합니다.
"""

import json
import math
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams
from sentence_transformers import SentenceTransformer

# 경로 및 컬렉션 설정
NPZ_PATH = "data/export/vectors.npz"
CHUNKS_PATH = "data/export/chunks.json"
COLLECTION_NAME = "law_kb"
EMBED_MODEL = "nlpai-lab/KURE-v1"


def get_sparse_vector(text: str, model: SentenceTransformer) -> dict:
    """BGE-M3 구조의 토크나이저 가중치로 Sparse 벡터를 생성합니다."""
    tokens = model.tokenizer.tokenize(text)
    token_ids = model.tokenizer.convert_tokens_to_ids(tokens)
    sparse_dict = {}
    for t_id in token_ids:
        sparse_dict[str(t_id)] = sparse_dict.get(str(t_id), 0.0) + 1.0
    norm = math.sqrt(sum(v ** 2 for v in sparse_dict.values()))
    return {int(k): v / norm for k, v in sparse_dict.items()}


def main():
    client = QdrantClient(host="localhost", port=6333)

    # 하이브리드 검색을 위한 모델 로드 (Sparse 가중치 계산용)
    print(f"📦 토크나이저 활용을 위한 모델 로드: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print("💾 로컬 임베딩 파일 및 청크 로딩 중...")
    # npz 데이터 로드
    npz_data = np.load(NPZ_PATH, allow_pickle=True)
    dense_vectors = npz_data["vectors"]
    chunk_ids = npz_data["chunk_ids"]

    # 원본 청크 데이터 로드
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # 1. 기존 컬렉션이 있다면 초기화 후 재생성 (Dense + Sparse 듀얼 벡터 레이아웃 빌드)
    print(f"🧹 Qdrant 컬렉션 '{COLLECTION_NAME}' 생성 중...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=len(dense_vectors[0]),  # KURE-v1 배포 차원 규격 (1024)
                distance=Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=None
            )
        }
    )

    # 2. 하이브리드 포맷으로 데이터 패킹 후 Upsert
    print("🚀 Qdrant DB로 하이브리드 벡터 마이그레이션 시작...")

    points = []
    for idx, c_id in enumerate(chunk_ids):
        # 원본 json과 매칭하여 페이로드 메타데이터 추출
        matched_chunk = next((c for c in chunks if c["chunk_id"] == c_id), {})
        text_content = matched_chunk.get("text", "")

        # 동적 Sparse 가중치 벡터 연산
        sparse_vec = get_sparse_vector(text_content, model)

        # Qdrant 저장 규격 빌드
        from qdrant_client.models import PointStruct, SparseVector
        points.append(
            PointStruct(
                id=idx,  # 정수형 고유 ID 지정
                vector={
                    "dense": dense_vectors[idx].tolist(),
                    "sparse": SparseVector(indices=list(sparse_vec.keys()), values=list(sparse_vec.values()))
                },
                payload={
                    "chunk_id": str(c_id),
                    "law_name": matched_chunk.get("law_name", "지방자치단체를 당사자로 하는 계약에 관한 법률"),
                    "article_number": matched_chunk.get("article_number", ""),
                    "article_title": matched_chunk.get("article_title", ""),
                    "chunk_text": text_content,
                    "source_full": matched_chunk.get("source_full", f"지방계약법 {matched_chunk.get('article_number', '')}"),
                    "is_risk_ref": True,  # 평가를 위해 True 마킹 보존
                    "risk_ids": matched_chunk.get("risk_ids", []),
                    "risk_names": matched_chunk.get("risk_names", [])
                }
            )
        )

    # 대량 데이터 일괄 적재
    client.upsert(collection_name=COLLECTION_NAME, points=points)

    # 최종 검증 확인
    count = client.count(collection_name=COLLECTION_NAME)
    print(f"\n✅ 마이그레이션 완료! law_kb 컬렉션에 {count.count}개 청크가 성공적으로 빌드되었습니다.")


if __name__ == "__main__":
    main()