from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List, Tuple


def normalize_compare_text(text: str) -> str:
    """비교 전용 정규화: 공백/괄호/특수문자를 제거해서 표기 차이로 인한 오탐을 줄인다."""
    text = text or ""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[()（）\[\]【】ㆍ·.,:;]", "", text)
    return text.strip()


def clean_text(text: str) -> str:
    """
    소제목 위치 탐색 전에, 위치 탐색을 방해하는 노이즈를 제거한다.
    특히 '목차' 영역에 소제목 텍스트가 미리 나열되어 있으면,
    본문보다 목차 쪽 위치를 먼저 찾아버리는 문제가 생기므로 이를 최대한 걸러낸다.
    """
    text = text or ""
    text = text.replace("\u00a0", " ")

    # "소제목 ..... 12" 같이 제목 뒤에 점선/가운데점 + 페이지번호가 붙는 목차 줄 제거
    text = re.sub(r"^.{1,60}[.\u2026·]{3,}\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
    # 단독 숫자 줄 (페이지 번호로 추정)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*-\s*\d+\s*-\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*Page\s*\d+\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    # "목차" 단독 줄 제거
    text = re.sub(r"^\s*목\s*차\s*$", "", text, flags=re.MULTILINE)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def similarity(a: str, b: str) -> float:
    """difflib 기반 유사도 (0~1). 내부에서 정규화 후 비교한다."""
    return SequenceMatcher(
        None,
        normalize_compare_text(a),
        normalize_compare_text(b),
    ).ratio()


def normalize_with_map(text: str) -> Tuple[str, List[int]]:
    """
    normalize_compare_text와 동일한 정규화를 하되, 정규화된 각 글자가
    원본 텍스트의 몇 번째 인덱스였는지 매핑을 같이 반환한다.

    이걸로 '정규화된 텍스트에서 찾은 위치'를 '줄바꿈이 살아있는 원본 텍스트의
    위치'로 되돌릴 수 있어서, 문단 구분을 유지한 채로 섹션 구간을 잘라낼 수 있다.
    """
    text = text or ""
    text = text.replace("\u00a0", " ")
    strip_chars = set("()（）[]【】ㆍ·.,:;")

    out_chars = []
    index_map = []
    for i, ch in enumerate(text):
        if ch.isspace() or ch in strip_chars:
            continue
        out_chars.append(ch)
        index_map.append(i)

    return "".join(out_chars), index_map


def containment_ratio(fragment: str, full_text: str) -> float:
    """
    fragment(원본 문단 한 조각)가 full_text(파싱 결과 content 등) 안에
    얼마나 그대로 들어있는지 비율. fragment 길이를 분모로 삼기 때문에,
    full_text가 다른 내용과 섞여 길어도 fragment 자체의 포함 여부를 정확히 잰다.
    (양쪽 길이를 더해서 나누는 일반 유사도와 달리, '부분 포함' 여부를 볼 때 더 정확함)
    """
    frag_norm = normalize_compare_text(fragment)
    text_norm = normalize_compare_text(full_text)
    if not frag_norm:
        return 1.0
    matcher = SequenceMatcher(None, frag_norm, text_norm, autojunk=False)
    matched = sum(m.size for m in matcher.get_matching_blocks())
    return matched / len(frag_norm)


def split_paragraphs(text: str) -> List[str]:
    """빈 줄 기준으로 문단을 나누고, 빈 줄이 없으면 단순 줄바꿈 기준으로 나눈다."""
    text = text or ""
    blocks = re.split(r"\n\s*\n", text)
    if len(blocks) <= 1:
        blocks = re.split(r"\n", text)
    return [b.strip() for b in blocks if b.strip()]
