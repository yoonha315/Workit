"""
Workit - RAG Retrieval 평가
Recall@5, Recall@10, MRR 측정

골드셋: taxonomy 9개 리스크 → 쿼리 + 정답 조문 매핑
실행:
    pip install qdrant-client sentence-transformers
    python workit_rag_eval.py
"""

import json
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer


# ──────────────────────────────────────────
# 0. 설정
# ──────────────────────────────────────────
QDRANT_PATH = "./qdrant_storage"
COLLECTION  = "law_kb"
EMBED_MODEL = "BAAI/bge-m3"
TOP_K       = 10        # Recall@5, @10 둘 다 커버
MIN_SCORE   = 0.0       # 평가 시에는 threshold 끄고 순위만 봄


# ──────────────────────────────────────────
# 1. 골드셋
#    형식: 쿼리 → 정답 source_full 목록
#    정답이 여러 조문인 경우 (RISK_003, 005, 008) 전부 포함
#    → 하나라도 top-K 안에 있으면 hit로 처리
# ──────────────────────────────────────────
GOLD_SET = [
    {
        "risk_id":   "RISK_001",
        "risk_name": "지연배상금 상한 미설정",
        "query":     "지연배상금 총액의 한도가 설정되지 않은 계약 조항",
        "gold_sources": [
            "지방계약법 시행령 제90조",
        ],
    },
    {
        "risk_id":   "RISK_002",
        "risk_name": "지연배상금률 과다 설정",
        "query":     "지연배상금률이 법정 기준을 초과하여 과도하게 설정된 경우",
        "gold_sources": [
            "지방계약법 시행규칙 제75조",
        ],
    },
    {
        "risk_id":   "RISK_003",
        "risk_name": "대금 지급 기한 미설정",
        "query":     "용역 대금 지급 기한이 명시되지 않거나 불명확한 계약 조항",
        "gold_sources": [
            "지방계약법 제18조",
            "지방계약법 시행령 제67조",
        ],
    },
    {
        "risk_id":   "RISK_004",
        "risk_name": "합의 없는 일방적 과업 추가",
        "query":     "발주자가 일방적으로 과업을 추가할 수 있도록 한 계약 조항",
        "gold_sources": [
            "지방자치단체 용역계약 일반조건 제6절 제1항 나",
            "소프트웨어 진흥법 제50조",
        ],
    },
    {
        "risk_id":   "RISK_005",
        "risk_name": "추가 과업 비용 미지급",
        "query":     "추가 과업 수행 시 별도 대가를 지급하지 않는 계약 조항",
        "gold_sources": [
            "지방자치단체 용역계약 일반조건 제6절 제1항 라",
            "지방자치단체 용역계약 일반조건 제6절 제1항 마",
            "지방계약법 시행령 제74조",
        ],
    },
    {
        "risk_id":   "RISK_006",
        "risk_name": "갑의 일방적 해지권",
        "query":     "발주자가 일방적으로 계약을 해지할 수 있으며 기수행 대가 지급 규정이 없는 조항",
        "gold_sources": [
            "지방자치단체 용역계약 일반조건 제7절 제4항 다",
        ],
    },
    {
        "risk_id":   "RISK_007",
        "risk_name": "을의 해제권 배제/제한",
        "query":     "수급인의 계약 해지권이 배제되거나 제한된 계약 조항",
        "gold_sources": [
            "지방자치단체 용역계약 일반조건 제7절 제5항 가",
        ],
    },
    {
        "risk_id":   "RISK_008",
        "risk_name": "계약금액 조정 없는 과업변경",
        "query":     "과업 내용이 변경되어도 계약금액을 조정하지 않는 계약 조항",
        "gold_sources": [
            "지방자치단체 용역계약 일반조건 제6절 제1항 라",
            "지방계약법 제22조",
            "지방계약법 시행령 제74조",
        ],
    },
    {
        "risk_id":   "RISK_009",
        "risk_name": "손해배상 범위 일방적 제한",
        "query":     "귀책 사유와 무관하게 수급인이 모든 손해를 부담하도록 한 조항",
        "gold_sources": [
            "지방자치단체 용역계약 일반조건 제8절 제7항 가",
        ],
    },
]


# ──────────────────────────────────────────
# 2. 단일 쿼리 검색
# ──────────────────────────────────────────
def retrieve(
    query: str,
    client: QdrantClient,
    model: SentenceTransformer,
    top_k: int = TOP_K,
    risk_only: bool = True,         # is_risk_ref=True 조문만 검색
) -> list[dict]:
    """
    쿼리 → Qdrant 검색 → source_full 목록 반환 (score 순)
    """
    vector = model.encode(query).tolist()

    search_filter = None
    if risk_only:
        search_filter = Filter(
            must=[FieldCondition(key="is_risk_ref", match=MatchValue(value=True))]
        )

    results = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        query_filter=search_filter,
        limit=top_k,
    ).points

    return [
        {
            "source_full": p.payload.get("source_full", ""),
            "score":       round(float(p.score), 4),
            "risk_names":  p.payload.get("risk_names", []),
        }
        for p in results
    ]


# ──────────────────────────────────────────
# 3. 평가 지표 계산
# ──────────────────────────────────────────
@dataclass
class EvalResult:
    risk_id:      str
    risk_name:    str
    query:        str
    gold_sources: list[str]
    retrieved:    list[dict]    # top-10 검색 결과
    hit_rank:     int           # 첫 번째 정답이 몇 번째에 등장했는지 (없으면 0)
    hit_at_5:     bool
    hit_at_10:    bool
    rr:           float         # Reciprocal Rank


def first_hit_rank(retrieved: list[dict], gold_sources: list[str]) -> int:
    """
    retrieved 목록에서 gold_sources 중 하나가 처음 등장하는 순위 반환.
    없으면 0 반환.
    """
    for rank, item in enumerate(retrieved, start=1):
        if item["source_full"] in gold_sources:
            return rank
    return 0


def evaluate_single(
    sample: dict,
    client: QdrantClient,
    model: SentenceTransformer,
) -> EvalResult:
    retrieved = retrieve(sample["query"], client, model, top_k=TOP_K)
    rank      = first_hit_rank(retrieved, sample["gold_sources"])

    return EvalResult(
        risk_id      = sample["risk_id"],
        risk_name    = sample["risk_name"],
        query        = sample["query"],
        gold_sources = sample["gold_sources"],
        retrieved    = retrieved,
        hit_rank     = rank,
        hit_at_5     = 1 <= rank <= 5,
        hit_at_10    = 1 <= rank <= 10,
        rr           = (1 / rank) if rank > 0 else 0.0,
    )


def compute_metrics(results: list[EvalResult]) -> dict:
    n         = len(results)
    recall_5  = sum(r.hit_at_5  for r in results) / n
    recall_10 = sum(r.hit_at_10 for r in results) / n
    mrr       = sum(r.rr        for r in results) / n

    return {
        "n":          n,
        "Recall@5":   round(recall_5,  4),
        "Recall@10":  round(recall_10, 4),
        "MRR":        round(mrr,       4),
    }


# ──────────────────────────────────────────
# 4. 리포트 출력
# ──────────────────────────────────────────
def print_report(results: list[EvalResult], metrics: dict) -> None:
    print("\n" + "=" * 65)
    print("Workit RAG Retrieval 평가 결과")
    print("=" * 65)

    for r in results:
        hit_icon = "✅" if r.hit_at_5 else ("🟡" if r.hit_at_10 else "❌")
        rank_str = f"rank={r.hit_rank}" if r.hit_rank > 0 else "미검색"

        print(f"\n{hit_icon} [{r.risk_id}] {r.risk_name}")
        print(f"   쿼리   : {r.query}")
        print(f"   정답   : {', '.join(r.gold_sources)}")
        print(f"   결과   : {rank_str}  |  RR={r.rr:.3f}")

        print(f"   Top-{len(r.retrieved)} 검색 결과:")
        for i, item in enumerate(r.retrieved, 1):
            marker = " ← HIT" if item["source_full"] in r.gold_sources else ""
            print(f"     {i:2}. [{item['score']:.4f}] {item['source_full']}{marker}")

    print("\n" + "─" * 65)
    print(f"{'지표':<15} {'값':>10}")
    print("─" * 65)
    print(f"{'Recall@5':<15} {metrics['Recall@5']:>10.4f}  ({sum(r.hit_at_5 for r in results)}/{metrics['n']})")
    print(f"{'Recall@10':<15} {metrics['Recall@10']:>10.4f}  ({sum(r.hit_at_10 for r in results)}/{metrics['n']})")
    print(f"{'MRR':<15} {metrics['MRR']:>10.4f}")
    print("─" * 65)

    # 실패 케이스 요약
    failed = [r for r in results if not r.hit_at_10]
    if failed:
        print(f"\n❌ top-10 미검색 ({len(failed)}건):")
        for r in failed:
            print(f"   - [{r.risk_id}] {r.risk_name}")
            print(f"     정답: {', '.join(r.gold_sources)}")
    else:
        print("\n🎉 전 항목 top-10 내 검색 성공")


# ──────────────────────────────────────────
# 5. JSON 저장
# ──────────────────────────────────────────
def save_results(results: list[EvalResult], metrics: dict, path: str = "eval_results.json") -> None:
    output = {
        "metrics": metrics,
        "details": [
            {
                "risk_id":      r.risk_id,
                "risk_name":    r.risk_name,
                "query":        r.query,
                "gold_sources": r.gold_sources,
                "hit_rank":     r.hit_rank,
                "hit_at_5":     r.hit_at_5,
                "hit_at_10":    r.hit_at_10,
                "rr":           round(r.rr, 4),
                "top10":        r.retrieved,
            }
            for r in results
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n💾 상세 결과 저장: {path}")


# ──────────────────────────────────────────
# 6. 메인
# ──────────────────────────────────────────
def main() -> None:
    print("=" * 65)
    print("Workit RAG Retrieval 평가")
    print(f"골드셋: {len(GOLD_SET)}개 리스크 | TOP_K={TOP_K}")
    print("=" * 65)

    print(f"\n📦 모델 로드: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    client = QdrantClient(path=QDRANT_PATH)
    count  = client.count(collection_name=COLLECTION)
    print(f"📚 Qdrant KB: {count.count}개 청크")

    print("\n🔍 평가 시작...")
    results = [evaluate_single(sample, client, model) for sample in GOLD_SET]

    metrics = compute_metrics(results)
    print_report(results, metrics)
    save_results(results, metrics)


if __name__ == "__main__":
    main()