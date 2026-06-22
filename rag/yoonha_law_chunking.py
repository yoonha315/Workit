"""
═══════════════════════════════════════════════════════════════════
STEP 1 — PyCharm에서 실행
역할: 법령 JSON → 청크 생성 → chunks.json 저장
═══════════════════════════════════════════════════════════════════

실행:
    python rag/yoonha_law_chunking.py

출력:
    data/export/chunks.json   ← Colab에 업로드할 파일
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path


# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────
LAW_REFS_PATH  = Path("data/law_refs.json")
STRUCTURED_DIR = Path("data/structured")
EXPORT_DIR     = Path("data/export")

LAW_FILES = [
    {"filename": "지방계약법.json",                                "law_name": "지방계약법",                    "prefix": "LCA"},
    {"filename": "지방계약법_시행령.json",                          "law_name": "지방계약법 시행령",              "prefix": "LCAE"},
    {"filename": "지방계약법_시행규칙.json",                        "law_name": "지방계약법 시행규칙",            "prefix": "LCAR"},
    {"filename": "소프트웨어_진흥법.json",                          "law_name": "소프트웨어 진흥법",              "prefix": "SWPA"},
    {"filename": "지방회계법.json",                                 "law_name": "지방회계법",                    "prefix": "LARA"},
    {"filename": "지방자치단체 용역계약 일반조건 (행안부 예규).json", "law_name": "지방자치단체 용역계약 일반조건",  "prefix": "PYG"},
]

RISK_CATEGORIES = {
    "지체상금", "이행보증", "대금지급", "검사",
    "하자담보", "계약금액조정", "과업내용", "계약해제해지",
    "필수기재", "부당특약",
}


# ──────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────
def now() -> str:
    return datetime.now().strftime("%H:%M:%S")

def elapsed(start: float) -> str:
    s = time.time() - start
    return f"{s:.1f}초" if s < 60 else f"{int(s//60)}분 {int(s%60)}초"


# ──────────────────────────────────────────
# law_refs.json 로드
# ──────────────────────────────────────────
def load_law_refs(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────
# 법령 JSON 로드
# ──────────────────────────────────────────
def load_law_json(filepath: Path) -> list[dict]:
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    return [a for a in data.get("articles", []) if a.get("text", "").strip()]


# ──────────────────────────────────────────
# chunk_id 생성 (의N suffix 포함 수정)
# ──────────────────────────────────────────
def _make_chunk_id(prefix: str, article_id: str, hierarchy: dict | None = None) -> str:
    def extract_num(s: str) -> str:
        """제N조 → N, 제N조의M → N_M"""
        m = re.search(r"제\s*(\d+)(?:조의(\d+))?", s)
        if m:
            return f"{m.group(1)}_{m.group(2)}" if m.group(2) else m.group(1)
        return s

    if hierarchy:
        if prefix == "PYG":
            parts = []
            if "절" in hierarchy:
                parts.append(extract_num(hierarchy["절"]))
            if "항" in hierarchy:
                parts.append(extract_num(hierarchy["항"]))
            return f"{prefix}_{'_'.join(parts)}" if parts else f"{prefix}_{article_id}"
        else:
            parts = []
            for key in ("조", "항", "호"):
                if key in hierarchy:
                    val = hierarchy[key]
                    if key == "호":
                        num = re.search(r"제\s*(\d+)", val)
                        parts.append(num.group(1) if num else val.strip())
                    else:
                        parts.append(extract_num(val))
            return f"{prefix}_{'_'.join(parts)}" if parts else f"{prefix}_{article_id}"

    if prefix == "PYG":
        nums = re.findall(r"제(\d+)", article_id)
        return f"{prefix}_{'_'.join(nums[:2])}" if nums else f"{prefix}_{article_id}"
    else:
        # 제N조의M → N_M, 제N조 → N
        m = re.search(r"제(\d+)조(?:의(\d+))?", article_id)
        if m:
            base = m.group(1)
            suffix = f"_{m.group(2)}" if m.group(2) else ""
            # 항/호 추가
            sub = re.findall(r"제(\d+)[항호]", article_id)
            sub_str = f"_{'_'.join(sub)}" if sub else ""
            return f"{prefix}_{base}{suffix}{sub_str}"

        # fallback
        nums = re.findall(r"\d+", article_id)
        return f"{prefix}_{'_'.join(nums)}" if nums else f"{prefix}_{article_id}"


# ──────────────────────────────────────────
# 청크 생성
# ──────────────────────────────────────────
def build_chunks(law_refs: dict) -> list[dict]:
    ref_index = {chunk_id: meta for chunk_id, meta in law_refs.items()}
    chunks = []
    t0 = time.time()

    print(f"\n[{now()}] 📂 법령 JSON 로드 + 청크 생성")
    for law in LAW_FILES:
        filepath = STRUCTURED_DIR / law["filename"]
        if not filepath.exists():
            print(f"  ⚠️  파일 없음: {law['filename']}")
            continue

        t_file = time.time()
        articles = load_law_json(filepath)
        tagged_count = 0

        for article in articles:
            article_id = article.get("article_id", "")
            text       = article.get("text", "").strip()
            if not text:
                continue

            hierarchy  = article.get("hierarchy")
            chunk_id   = _make_chunk_id(law["prefix"], article_id, hierarchy)
            meta       = ref_index.get(chunk_id, {})
            category   = meta.get("category", "")
            is_risk    = category in RISK_CATEGORIES
            if is_risk:
                tagged_count += 1

            chunks.append({
                "chunk_id":    chunk_id,
                "law_name":    law["law_name"],
                "article_id":  article_id,
                "category":    category,
                "article":     meta.get("article", ""),
                "is_risk_ref": is_risk,
                "text":        text,
            })

        print(f"  ✅ [{now()}] {law['law_name']} — {len(articles)}개 조문 | 태깅: {tagged_count}개 | {elapsed(t_file)}")

    total_tagged = sum(1 for c in chunks if c["is_risk_ref"])
    print(f"\n  → 전체 {len(chunks)}개 청크 | is_risk_ref=True: {total_tagged}개 | {elapsed(t0)}")

    # law_refs에 있는데 청크에 없는 chunk_id 경고
    chunk_id_set = {c["chunk_id"] for c in chunks}
    missing = [cid for cid in ref_index if cid not in chunk_id_set]
    if missing:
        print(f"\n  ⚠️  law_refs에 있지만 청크 미생성: {missing}")

    return chunks


# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────
def main():
    t_total = time.time()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  STEP 1 — 청크 생성 & 내보내기  [{now()}]")
    print("=" * 60)

    print(f"\n[{now()}] 📋 law_refs.json 로드")
    law_refs = load_law_refs(LAW_REFS_PATH)
    print(f"  → {len(law_refs)}개 메타 등록 조문")

    chunks = build_chunks(law_refs)

    out_path = EXPORT_DIR / "chunks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\n[{now()}] 💾 저장 완료: {out_path}  ({size_mb:.1f} MB)")
    print(f"  → 이 파일을 Google Colab에 업로드하세요")
    print(f"\n총 소요: {elapsed(t_total)}")
    print("=" * 60)


if __name__ == "__main__":
    main()