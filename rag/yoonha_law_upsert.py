"""
Workit - law_kb Qdrant upsert 스크립트 (Hybrid: Dense + Sparse)
파일명: yoonha_law_upsert.py
위치:   Workit/rag/yoonha_law_upsert.py

데이터:
  data/export/chunks.json          → payload
  data/export/vectors.npz          → dense 벡터 (N, 1024) float16
  data/export/sparse_weights.json  → BGE-M3 sparse lexical weights

실행:
  python rag/yoonha_law_upsert.py
"""

from __future__ import annotations

import json
import math
import numpy as np
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION  = "law_kb"
VECTOR_DIM  = 1024
BATCH_SIZE  = 64

_THIS_DIR        = Path(__file__).resolve().parent
_DATA_DIR        = _THIS_DIR.parent / "data" / "export"

CHUNKS_PATH      = _DATA_DIR / "chunks.json"
VECTORS_PATH     = _DATA_DIR / "vectors.npz"
SPARSE_PATH      = _DATA_DIR / "sparse_weights.json"


# ──────────────────────────────────────────
# sparse weight 변환
# ──────────────────────────────────────────
def to_sparse_vector(lexical_weights: dict) -> SparseVector:
    """
    BGE-M3 lexical_weights {token_str: weight} →
    Qdrant SparseVector {indices: [...], values: [...]}

    token_str은 문자열이므로 정수 인덱스로 변환 불가 →
    문자열 자체를 hash로 변환하여 인덱스로 사용.
    (retrieval 시 동일 모델로 동일 hash 생성하므로 일관성 유지)
    """
    indices = []
    values  = []
    for token_str, weight in lexical_weights.items():
        # token_str → 양의 정수 인덱스 (hash & 0x7FFFFFFF)
        idx = hash(token_str) & 0x7FFFFFFF
        indices.append(idx)
        values.append(float(weight))
    return SparseVector(indices=indices, values=values)


# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────
def main() -> None:
    print("=" * 55)
    print("Workit law_kb — Qdrant Hybrid Upsert")
    print("=" * 55)

    # ── 청크 로드 ─────────────────────────
    print(f"\n📂 chunks.json 로드: {CHUNKS_PATH}")
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks: list[dict] = json.load(f)

    # chunk_id 기준 중복 제거 (마지막 항목 유지)
    chunk_map: dict[str, dict] = {}
    for c in chunks:
        chunk_map[c["chunk_id"]] = c
    print(f"   원본 {len(chunks)}개 → chunk_id 중복 제거 후 {len(chunk_map)}개")

    # 텍스트 기준 추가 중복 제거
    # 동일 텍스트가 다른 chunk_id로 여러 번 파싱된 경우 제거
    # (예: LCAE_90_1_3, LCAE_90_1_7, LCAE_90_1_11 — 호 오파싱 산물)
    seen_texts: set[str] = set()
    deduped: dict[str, dict] = {}
    for cid, chunk in chunk_map.items():
        t = chunk.get("text", "").strip()
        if t not in seen_texts:
            seen_texts.add(t)
            deduped[cid] = chunk
    print(f"   텍스트 중복 제거 후 {len(deduped)}개")
    chunk_map = deduped

    # ── 벡터 로드 ─────────────────────────
    print(f"\n📂 vectors.npz 로드: {VECTORS_PATH}")
    npz       = np.load(VECTORS_PATH)
    vectors   = npz["vectors"].astype(np.float32)
    chunk_ids = npz["chunk_ids"].tolist()
    print(f"   벡터 shape: {vectors.shape}")

    # dense 벡터 맵 (chunk_id → vector)
    # 중복 chunk_id는 마지막 등장한 벡터 사용
    id_to_dense: dict[str, list[float]] = {}
    for cid, vec in zip(chunk_ids, vectors):
        id_to_dense[cid] = vec.tolist()

    # ── sparse 로드 ───────────────────────
    use_sparse = SPARSE_PATH.exists()
    id_to_sparse: dict[str, dict] = {}

    if use_sparse:
        print(f"\n📂 sparse_weights.json 로드: {SPARSE_PATH}")
        with open(SPARSE_PATH, encoding="utf-8") as f:
            sparse_list: list[dict] = json.load(f)

        # chunk_ids 순서와 sparse_list 순서가 동일하다고 가정
        for cid, sw in zip(chunk_ids, sparse_list):
            id_to_sparse[cid] = sw
        print(f"   sparse 벡터: {len(id_to_sparse)}개")
    else:
        print(f"\n⚠️  sparse_weights.json 없음 → dense 단독 upsert")

    # ── upsert 대상 확정 ──────────────────
    common_ids = [cid for cid in chunk_map if cid in id_to_dense]
    print(f"\n   upsert 대상: {len(common_ids)}개")

    # ── Qdrant 컬렉션 준비 ────────────────
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        print(f"\n⚠️  컬렉션 '{COLLECTION}' 이미 존재 → 재생성합니다.")
        client.delete_collection(COLLECTION)

    if use_sparse:
        # Dense + Sparse 하이브리드 컬렉션
        from qdrant_client.models import SparseVectorParams
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                "dense": VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )
        print(f"✅ 컬렉션 '{COLLECTION}' 생성 완료 (Dense + Sparse)")
    else:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        print(f"✅ 컬렉션 '{COLLECTION}' 생성 완료 (Dense only)")

    # ── 배치 upsert ───────────────────────
    print(f"\n⬆️  upsert 시작 (batch_size={BATCH_SIZE})...")
    points: list[PointStruct] = []

    for i, cid in enumerate(common_ids):
        chunk   = chunk_map[cid]
        payload = {
            "chunk_id"   : cid,
            "law_name"   : chunk.get("law_name",   ""),
            "article_id" : chunk.get("article_id", ""),
            "article"    : chunk.get("article",    ""),
            "category"   : chunk.get("category",   ""),
            "is_risk_ref": bool(chunk.get("is_risk_ref", False)),
            "chunk_text" : chunk.get("text",        ""),
            # source_full: evaluation 파일 gold_sources 매칭용
            "source_full": chunk.get("article",    ""),
        }

        if use_sparse and cid in id_to_sparse:
            sparse_vec = to_sparse_vector(id_to_sparse[cid])
            point = PointStruct(
                id=i,
                vector={
                    "dense" : id_to_dense[cid],
                    "sparse": sparse_vec,
                },
                payload=payload,
            )
        else:
            point = PointStruct(
                id=i,
                vector=id_to_dense[cid],
                payload=payload,
            )

        points.append(point)

        if len(points) == BATCH_SIZE:
            client.upsert(collection_name=COLLECTION, points=points)
            print(f"   [{i + 1}/{len(common_ids)}] upsert...", end="\r")
            points = []

    if points:
        client.upsert(collection_name=COLLECTION, points=points)

    # ── 확인 ─────────────────────────────
    count = client.count(collection_name=COLLECTION)
    print(f"\n✅ 완료: {count.count}개 포인트 저장됨")

    risk_count = client.count(
        collection_name=COLLECTION,
        count_filter={"must": [{"key": "is_risk_ref", "match": {"value": True}}]},
    )
    print(f"   is_risk_ref=True: {risk_count.count}개")


if __name__ == "__main__":
    main()