"""
Workit - law_kb Qdrant upsert 스크립트
파일명: yoonha_law_upsert.py
위치:   Workit/rag/yoonha_law_upsert.py

데이터:
  data/export/chunks.json   → payload
  data/export/vectors.npz   → dense 벡터 (609, 1024) float16

실행:
  python yoonha_law_upsert.py
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
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

_THIS_DIR   = Path(__file__).resolve().parent
_DATA_DIR   = _THIS_DIR.parent / "data" / "export"

CHUNKS_PATH  = _DATA_DIR / "chunks.json"
VECTORS_PATH = _DATA_DIR / "vectors.npz"


# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────
def main() -> None:
    print("=" * 55)
    print("Workit law_kb — Qdrant upsert")
    print("=" * 55)

    # ── 데이터 로드 ───────────────────────
    print(f"\n📂 chunks.json 로드: {CHUNKS_PATH}")
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks: list[dict] = json.load(f)

    # chunk_id → 마지막 항목으로 중복 제거
    chunk_map: dict[str, dict] = {}
    for c in chunks:
        chunk_map[c["chunk_id"]] = c
    print(f"   원본 {len(chunks)}개 → 중복 제거 후 {len(chunk_map)}개")

    print(f"\n📂 vectors.npz 로드: {VECTORS_PATH}")
    npz        = np.load(VECTORS_PATH)
    vectors    = npz["vectors"].astype(np.float32)   # float16 → float32
    chunk_ids  = npz["chunk_ids"].tolist()
    print(f"   벡터 shape: {vectors.shape}")

    # ── chunk_id 기준으로 벡터-페이로드 매핑 ──
    # 중복 chunk_id는 마지막 등장한 벡터 사용
    id_to_vec: dict[str, list[float]] = {}
    for cid, vec in zip(chunk_ids, vectors):
        id_to_vec[cid] = vec.tolist()

    # chunk_map 과 id_to_vec 교집합만 upsert
    common_ids = [cid for cid in chunk_map if cid in id_to_vec]
    print(f"   upsert 대상: {len(common_ids)}개")

    # ── Qdrant 컬렉션 준비 ────────────────
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        print(f"\n⚠️  컬렉션 '{COLLECTION}' 이미 존재 → 재생성합니다.")
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    print(f"✅ 컬렉션 '{COLLECTION}' 생성 완료")

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
        }
        points.append(PointStruct(id=i, vector=id_to_vec[cid], payload=payload))

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