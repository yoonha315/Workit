"""
Workit - 법령 문서 파싱 스크립트
input : C:/project/Workit/data/law/ 내 doc/docx 파일
output: C:/project/Workit/data/structured/ 내 JSON 파일

사용법:
    pip install python-docx pywin32
    python yoonha_law_parser.py
"""

import os
import re
import json
from pathlib import Path
from docx import Document
import win32com.client

# ─────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────
LAW_DIR    = Path("C:/project/Workit/data/law")
OUTPUT_DIR = Path("C:/project/Workit/data/structured")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────
# REF_ARTICLE & UPPER_LAW (필터링 기준)
# ─────────────────────────────────────────
REF_ARTICLE = [
    "제7절 제1항 가",
    "제8절 제4항 나",
    "제6절 제1항 가",
    "제6절 제1항 라",
    "제6절 제1항 마",
    "제7절 제4항 다",
    "제7절 제5항 가",
    "제8절 제7항 가",
    "제59조",
    "제75조",
]

UPPER_LAW = [
    "제90조",
    "제75조",
    "제27조",
    "제50조",
    "제59조",
    "제22조",
]

# ─────────────────────────────────────────
# 파일명 → document_type 매핑
# ─────────────────────────────────────────
FILE_META = {
    "지방자치단체 용역계약 일반조건 (행안부 예규)": {
        "document_type": "지방자치단체 용역계약 일반조건",
        "source": "행정안전부 예규",
        "is_ref_article_doc": True,
    },
    "지방계약법": {
        "document_type": "지방계약법",
        "source": "법률",
        "is_ref_article_doc": False,
    },
    "지방계약법_시행령": {
        "document_type": "지방계약법 시행령",
        "source": "대통령령",
        "is_ref_article_doc": False,
    },
    "지방계약법_시행규칙": {
        "document_type": "지방계약법 시행규칙",
        "source": "행정안전부령",
        "is_ref_article_doc": False,
    },
    "소프트웨어_진흥법": {
        "document_type": "소프트웨어 진흥법",
        "source": "법률",
        "is_ref_article_doc": False,
    },
    "지방회계법": {
        "document_type": "지방회계법",
        "source": "법률",
        "is_ref_article_doc": False,
    },
}

# ─────────────────────────────────────────
# 법령약자 매핑 (chunk_id 생성용)
# ─────────────────────────────────────────
DOC_TYPE_TO_PREFIX = {
    "지방계약법":                    "LCA",
    "지방계약법 시행령":              "LCAE",
    "지방계약법 시행규칙":            "LCAR",
    "소프트웨어 진흥법":              "SWPA",
    "지방회계법":                    "LARA",
    "지방자치단체 용역계약 일반조건":  "PYG",
}

# ─────────────────────────────────────────
# 유틸 함수
# ─────────────────────────────────────────
def read_docx(path: Path) -> list[tuple[str, str]]:
    suffix = path.suffix.lower()

    if suffix == ".docx":
        from docx.oxml.ns import qn
        from docx.table import Table as DocxTable
        doc = Document(str(path))
        lines = []
        for block in doc.element.body:
            tag = block.tag.split('}')[-1]
            if tag == 'p':
                text = ''.join(r.text for r in block.iter(qn('w:t'))).strip()
                if text:
                    lines.append(('p', text))
            elif tag == 'tbl':
                tbl = DocxTable(block, doc)
                for row in tbl.rows:
                    for cell in row.cells:
                        t = cell.text.strip()
                        if t:
                            lines.append(('tbl', t))
        return lines

    elif suffix == ".doc":
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        try:
            doc = word.Documents.Open(str(path.resolve()))
            lines = []
            for i in range(1, doc.Paragraphs.Count + 1):
                text = doc.Paragraphs(i).Range.Text.strip()
                if text:
                    lines.append(('p', text))
            doc.Close(False)
            return lines
        finally:
            word.Quit()
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {suffix}")


def find_meta(filename: str) -> dict | None:
    # 긴 키부터 매칭 (지방계약법_시행령이 지방계약법보다 먼저 매칭되도록)
    for key in sorted(FILE_META.keys(), key=len, reverse=True):
        if key in filename:
            return FILE_META[key]
    return None


def make_article_id(article_number: str) -> str:
    return re.sub(r"[\s/·]", "_", article_number).strip("_")


# ─────────────────────────────────────────
# chunk_id 생성 — {법령약자}_{조}_{항}_{호}
# ─────────────────────────────────────────
def make_chunk_id(prefix: str, jo: int, hang: int | None = None, ho: int | None = None, jo_ui: int | None = None) -> str:
    """
    규칙: {prefix}_{조}[_{조의N}][_{항}][_{호}]
    예:   LCAE_90_3  /  LCAR_70_1_6  /  LCA_18  /  SWPA_50
    """
    parts = [str(jo)]
    if jo_ui is not None:
        parts.append(str(jo_ui))
    if hang is not None:
        parts.append(str(hang))
    if ho is not None:
        parts.append(str(ho))
    return f"{prefix}_{'_'.join(parts)}"


# ─────────────────────────────────────────
# 파서 1: 용역계약 일반조건 (절/항/호 구조) — 기존과 동일
# ─────────────────────────────────────────
def parse_yongye(lines: list[tuple[str, str]], prefix: str = "PYG") -> list[dict]:
    articles = []
    cur_section = None
    cur_clause  = None
    cur_item    = None
    buf = []
    seen_section_nums = []

    section_pat = re.compile(r"^제\s*(\d+)\s*절")
    clause_pat  = re.compile(r"^\s*(\d+)\s*\.")
    item_pat    = re.compile(r"^\s*([가나다라마바사아자차카타파하])\s*\.")

    HANGUL_ORDER = "가나다라마바사아자차카타파하"

    def flush():
        if cur_section and cur_clause and cur_item and buf:
            an = f"제{cur_section}절 제{cur_clause}항 {cur_item}"
            # chunk_id: PYG_{절}_{항} (호는 생략 — 예규 구조상 절+항이 최소 단위)
            chunk_id = f"{prefix}_{cur_section}_{cur_clause}"
            articles.append({
                "chunk_id":      chunk_id,
                "article_id":    make_article_id(an),
                "article_number": an,
                "text":          " ".join(buf),
                "hierarchy": {
                    "절": f"제{cur_section}절",
                    "항": f"제{cur_clause}항",
                    "호": cur_item,
                }
            })

    for typ, text in lines:
        sm = section_pat.match(text) if typ == 'tbl' else None
        cm = clause_pat.match(text) if typ == 'p' else None
        im = item_pat.match(text) if typ == 'p' else None

        if sm:
            flush(); buf = []
            raw_num = sm.group(1)
            count = seen_section_nums.count(raw_num)
            cur_section = str(int(raw_num) + count)
            seen_section_nums.append(raw_num)
            cur_clause = None; cur_item = None
        elif cm and cur_section:
            flush(); buf = []
            cur_clause = cm.group(1)
            cur_item = None
        elif im and cur_clause:
            flush()
            buf = [text]
            cur_item = im.group(1)
        elif cur_item:
            buf.append(text)

    flush()

    seen_ids: dict[str, bool] = {}
    for a in articles:
        an = a["article_number"]
        if an in seen_ids:
            old_sec = int(re.search(r"제(\d+)절", an).group(1))
            new_an = an.replace(f"제{old_sec}절", f"제{old_sec + 1}절")
            a["article_number"] = new_an
            a["article_id"]     = make_article_id(new_an)
            a["hierarchy"]["절"] = f"제{old_sec + 1}절"
        else:
            seen_ids[an] = True

    return articles


# ─────────────────────────────────────────
# 파서 2: 일반 법령 — 항/호 단위 분리 (수정)
# ─────────────────────────────────────────
def parse_law(lines: list[tuple[str, str]], prefix: str) -> list[dict]:
    """
    일반 법령 파싱 — 제N조 단위로 묶은 뒤, 내부 ①②③ 항과 1.2.3. 호를 분리.

    chunk_id 규칙: {prefix}_{조}[_{조의N}][_{항}][_{호}]
      - 항만 있으면:  LCAE_90_1
      - 항+호 있으면: LCAR_70_1_6
      - 조만 있으면:  LCA_18
    """
    # ── Step 1: 제N조 단위로 텍스트 묶기 ──
    article_pat = re.compile(r"^(제\s*\d+\s*조(?:의\s*\d+)?)\s*[(\[〔]?([^)\]\)〕\n]*)[)\]\)〕]?")

    raw_articles: list[dict] = []
    cur_jo    = None
    cur_jo_ui = None
    cur_title = ""
    buf: list[str] = []

    def flush_jo():
        if cur_jo is not None and buf:
            raw_articles.append({
                "jo":    cur_jo,
                "jo_ui": cur_jo_ui,
                "title": cur_title,
                "text":  " ".join(buf),
            })

    for _, text in lines:
        m = article_pat.match(text)
        if m:
            flush_jo()
            buf = [text]
            raw_jo_str = re.sub(r"\s+", "", m.group(1))  # 제90조의3
            jo_m = re.match(r"제(\d+)조(?:의(\d+))?", raw_jo_str)
            cur_jo    = int(jo_m.group(1)) if jo_m else None
            cur_jo_ui = int(jo_m.group(2)) if jo_m and jo_m.group(2) else None
            cur_title = m.group(2).strip() if m.group(2) else ""
        else:
            buf.append(text)

    flush_jo()

    # ── Step 2: 조 내부 텍스트에서 항/호 분리 ──
    # 항 패턴: ①②③... (원문자) 또는 공백 후 숫자+점
    hang_pat = re.compile(r"[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮]")
    HANG_MAP = {c: i+1 for i, c in enumerate("①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮")}

    # 호 패턴: "1. ", "2. " 형태 (항 내부)
    ho_pat = re.compile(r"\s(\d{1,2})\.\s")

    articles: list[dict] = []

    for raw in raw_articles:
        jo    = raw["jo"]
        jo_ui = raw["jo_ui"]
        text  = raw["text"]

        # 항 분리
        # 원문자로 split
        hang_splits = re.split(r"([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮])", text)

        if len(hang_splits) <= 1:
            # 항 없음 → 조 단위 청크
            chunk_id = make_chunk_id(prefix, jo, jo_ui=jo_ui)
            articles.append({
                "chunk_id":      chunk_id,
                "article_id":    f"제{jo}조" + (f"의{jo_ui}" if jo_ui else ""),
                "article_number": f"제{jo}조" + (f"의{jo_ui}" if jo_ui else ""),
                "title":         raw["title"],
                "text":          text,
                "hierarchy":     {"조": f"제{jo}조" + (f"의{jo_ui}" if jo_ui else "")},
            })
            continue

        # 항 있음 → 항 단위로 분리
        # hang_splits: [조 서문, ①, 항1텍스트, ②, 항2텍스트, ...]
        i = 1
        while i < len(hang_splits) - 1:
            hang_char = hang_splits[i]
            hang_text = hang_splits[i + 1].strip() if i + 1 < len(hang_splits) else ""
            hang_num  = HANG_MAP.get(hang_char, i)

            # 호 분리 (항 텍스트 내부)
            ho_splits = re.split(r"\s(\d{1,2})\.\s", hang_text)

            if len(ho_splits) <= 1:
                # 호 없음 → 항 단위 청크
                chunk_id = make_chunk_id(prefix, jo, hang=hang_num, jo_ui=jo_ui)
                articles.append({
                    "chunk_id":      chunk_id,
                    "article_id":    f"제{jo}조" + (f"의{jo_ui}" if jo_ui else "") + f"제{hang_num}항",
                    "article_number": f"제{jo}조" + (f"의{jo_ui}" if jo_ui else "") + f"제{hang_num}항",
                    "title":         raw["title"],
                    "text":          hang_char + hang_text,
                    "hierarchy":     {
                        "조": f"제{jo}조" + (f"의{jo_ui}" if jo_ui else ""),
                        "항": f"제{hang_num}항",
                    },
                })
            else:
                # 호 있음 → 호 단위 청크
                # ho_splits: [항 서문, 호번호, 호텍스트, 호번호, 호텍스트, ...]
                j = 1
                while j < len(ho_splits) - 1:
                    ho_num  = int(ho_splits[j])
                    ho_text = ho_splits[j + 1].strip() if j + 1 < len(ho_splits) else ""
                    chunk_id = make_chunk_id(prefix, jo, hang=hang_num, ho=ho_num, jo_ui=jo_ui)
                    articles.append({
                        "chunk_id":      chunk_id,
                        "article_id":    f"제{jo}조" + (f"의{jo_ui}" if jo_ui else "") + f"제{hang_num}항제{ho_num}호",
                        "article_number": f"제{jo}조" + (f"의{jo_ui}" if jo_ui else "") + f"제{hang_num}항제{ho_num}호",
                        "title":         raw["title"],
                        "text":          f"{hang_char} {ho_splits[0].strip()} {ho_num}. {ho_text}",
                        "hierarchy":     {
                            "조": f"제{jo}조" + (f"의{jo_ui}" if jo_ui else ""),
                            "항": f"제{hang_num}항",
                            "호": f"제{ho_num}호",
                        },
                    })
                    j += 2

            i += 2

    return articles


# ─────────────────────────────────────────
# 필터링
# ─────────────────────────────────────────
def tag_article(article: dict, is_ref_doc: bool) -> dict:
    an = article.get("article_number", "")
    article["is_ref_article"] = is_ref_doc and any(ref in an for ref in REF_ARTICLE)
    article["is_upper_law"]   = any(ref in an for ref in UPPER_LAW)
    return article


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────
def process_file(path: Path):
    filename = path.stem
    meta = find_meta(filename)
    if meta is None:
        print(f"[SKIP] 메타 없음: {filename}")
        return

    print(f"[PARSE] {filename}")
    paragraphs = read_docx(path)
    prefix     = DOC_TYPE_TO_PREFIX.get(meta["document_type"], "UNK")

    if meta["is_ref_article_doc"]:
        articles = parse_yongye(paragraphs, prefix=prefix)
    else:
        articles = parse_law(paragraphs, prefix=prefix)

    articles = [tag_article(a, meta["is_ref_article_doc"]) for a in articles]

    result = {
        "document_type":      meta["document_type"],
        "source":             meta["source"],
        "filename":           path.name,
        "total_articles":     len(articles),
        "ref_article_count":  sum(1 for a in articles if a.get("is_ref_article")),
        "upper_law_count":    sum(1 for a in articles if a.get("is_upper_law")),
        "articles":           articles,
    }

    out_path = OUTPUT_DIR / f"{filename}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"  → 저장: {out_path}")
    print(f"  → 전체 조문: {len(articles)}개 | REF: {result['ref_article_count']}개 | UPPER: {result['upper_law_count']}개")

    # law_refs chunk_id 매칭 확인
    chunk_ids = {a["chunk_id"] for a in articles}
    print(f"  → 생성된 chunk_id 샘플: {list(chunk_ids)[:5]}")


def main():
    files = list(LAW_DIR.glob("*.doc")) + list(LAW_DIR.glob("*.docx"))
    if not files:
        print(f"[ERROR] {LAW_DIR} 에 파일이 없습니다.")
        return

    for f in sorted(files):
        process_file(f)

    print("\n✅ 완료! 결과물:", OUTPUT_DIR)


if __name__ == "__main__":
    main()