from __future__ import annotations

from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import Counter
import json
import re
import time

try:
    import pdfplumber
except ImportError:
    pdfplumber = None  # type: ignore


DELIVERABLE_REVIEW_PROMPT = """
너는 Workit 산출물 파싱 품질 AI 검수 에이전트다.
검수 대상 산출물을 먼저 아래 3개 계열 중 하나로 분류한다.

1. 계획/요구사항 계열
   - 제안요청서, 과업수행계획서, 사업수행계획서
   - 사업 개요, 추진 배경, 사업 범위, 추진 체계, 추진 일정, 요구사항, 산출물, 보안/품질/관리 항목을 중심으로 본다.

2. 결과/완료보고 계열
   - 최종완료보고서, 최종결과보고서, 사업추진결과보고서
   - 사업 개요, 추진 결과, 수행 내용, 산출물 결과, 테스트/검증 결과, 운영 계획, 성과, 결론을 중심으로 본다.

3. 테스트/검증 계열
   - 테스트계획서, 테스트결과보고서, 검수확인서
   - 테스트 목적, 범위, 환경, 항목, 절차, 결과, 결함, 조치 결과, 검수 확인을 중심으로 본다.

판정 기준:
- 원본/파싱 내용 유사도 0.98 이상이면 내용 보존은 PASS로 본다.
- 0.95 이상 0.98 미만이면 WARN, 0.95 미만이면 FAIL로 본다.
- 소제목 검수는 문서 계열에 맞는 기준만 적용한다.
- 다른 계열의 소제목이 없다는 이유만으로 FAIL 처리하지 않는다.
- 목차, 페이지 번호, 작성 지침, 양식 안내 문구는 핵심 내용 누락으로 과대 판단하지 않는다.
"""

DELIVERABLE_TYPE_SECTION_TITLES = {
    "planning_requirement": [
        "사업개요",
        "제안개요",
        "추진배경 및 필요성",
        "서비스 내용",
        "사업 범위",
        "시스템 현황",
        "사업추진 방안",
        "추진목표",
        "추진체계",
        "추진일정",
        "추진방안",
        "제안요청 내용",
        "제안요청 개요",
        "목표 시스템 개념도",
        "요구사항 총괄표",
        "상세 요구사항",
    ],
    "result_completion": [
        "사업개요",
        "사업 목적",
        "사업 내용",
        "추진 경과",
        "추진 결과",
        "수행 내용",
        "개발내용",
        "시스템 구성도",
        "산출물",
        "테스트 결과",
        "검증 결과",
        "운영계획",
        "성과",
        "결론",
    ],
    "test_verification": [
        "테스트 개요",
        "테스트 목적",
        "테스트 범위",
        "테스트 환경",
        "테스트 항목",
        "테스트 절차",
        "테스트 결과",
        "결함 내역",
        "조치 결과",
        "검수 결과",
        "검수 확인",
    ],
}

DEFAULT_DELIVERABLE_TYPE = "planning_requirement"
DEFAULT_DELIVERABLE_SECTION_TITLES = DELIVERABLE_TYPE_SECTION_TITLES[DEFAULT_DELIVERABLE_TYPE]

PASS_SIMILARITY_THRESHOLD = 0.98
WARN_SIMILARITY_THRESHOLD = 0.95

DELIVERABLE_PARENT_TITLE_CHILDREN = {
    "planning_requirement": {
        "사업개요": ["제안개요", "추진배경 및 필요성", "서비스 내용", "사업 범위"],
        "시스템 현황": ["현행 시스템 개요", "현행 시스템 현황"],
        "사업추진 방안": ["추진목표", "추진체계", "추진일정", "추진방안"],
        "제안요청 내용": ["제안요청 개요", "목표 시스템 개념도", "요구사항 총괄표", "상세 요구사항"],
    },
    "result_completion": {},
    "test_verification": {},
}

DELIVERABLE_TITLE_ALIASES = {
    # 공통 / 계획·요구사항 계열
    "사업개요": [
        "사업 개요", "사업 목적", "개요", "프로젝트 개요", "과업 개요"
    ],
    "제안개요": [
        "제안 개요", "제안 목적", "과업 개요", "사업 기본정보"
    ],
    "추진배경 및 필요성": [
        "추진배경", "추진 배경", "필요성", "사업 필요성", "도입 배경", "구축 배경"
    ],
    "서비스 내용": [
        "서비스내용", "서비스 범위", "제공 서비스", "주요 서비스", "서비스 구성"
    ],
    "사업 범위": [
        "사업범위", "범위 및 내용", "사업 내용", "과업 범위", "수행 범위", "구축 범위"
    ],
    "시스템 현황": [
        "시스템현황", "현행 시스템", "현행시스템 현황", "현황", "As-Is 현황"
    ],
    "사업추진 방안": [
        "사업 추진 방안", "추진 방안", "추진계획", "추진 계획", "수행 방안"
    ],
    "추진목표": [
        "추진 목표", "사업 목표", "구축 목표", "목표"
    ],
    "추진체계": [
        "추진 체계", "수행 체계", "사업 수행 체계", "조직 체계"
    ],
    "추진일정": [
        "추진 일정", "사업 일정", "일정 계획", "수행 일정", "프로젝트 일정"
    ],
    "추진방안": [
        "추진 방안", "추진계획", "추진 계획", "수행방안", "수행 방안"
    ],
    "제안요청 내용": [
        "제안요청내용", "요청 내용", "제안 요청 사항", "요청사항"
    ],
    "제안요청 개요": [
        "제안 요청 개요", "요청 개요", "요구 개요"
    ],
    "목표 시스템 개념도": [
        "목표시스템 개념도", "목표 시스템 구성도", "To-Be 시스템", "목표 구성도"
    ],
    "요구사항 총괄표": [
        "요구사항 목록", "요구사항 요약", "요구사항 총괄", "요구사항 현황", "요구사항 리스트"
    ],
    "상세 요구사항": [
        "상세요구사항", "요구사항 상세", "세부 요구사항", "요구사항 세부내용"
    ],

    # 결과/완료보고 계열
    "사업 목적": [
        "사업목적", "목적", "추진 목적"
    ],
    "사업 내용": [
        "사업내용", "수행 내용", "과업 내용", "주요 수행 내용"
    ],
    "추진 경과": [
        "추진경과", "수행 경과", "진행 경과", "사업 경과"
    ],
    "추진 결과": [
        "추진결과", "수행 결과", "사업 결과", "완료 결과"
    ],
    "수행 내용": [
        "수행내용", "개발 수행 내용", "주요 수행 내역"
    ],
    "개발내용": [
        "개발 내용", "구축 내용", "개발 범위", "구현 내용"
    ],
    "시스템 구성도": [
        "시스템구성도", "구성도", "시스템 아키텍처", "아키텍처 구성"
    ],
    "산출물": [
        "산출물 목록", "제출 산출물", "산출 정보", "산출정보"
    ],
    "테스트 결과": [
        "테스트결과", "시험 결과", "검증 결과", "테스트 수행 결과"
    ],
    "검증 결과": [
        "검증결과", "검수 결과", "확인 결과"
    ],
    "운영계획": [
        "운영 계획", "운영 방안", "유지관리 계획"
    ],
    "성과": [
        "사업 성과", "추진 성과", "주요 성과"
    ],
    "결론": [
        "종합 의견", "총평", "결과 요약", "마무리"
    ],

    # 테스트/검증 계열
    "테스트 개요": [
        "테스트개요", "시험 개요", "검증 개요"
    ],
    "테스트 목적": [
        "테스트목적", "시험 목적", "검증 목적"
    ],
    "테스트 범위": [
        "테스트범위", "시험 범위", "검증 범위"
    ],
    "테스트 환경": [
        "테스트환경", "시험 환경", "검증 환경"
    ],
    "테스트 항목": [
        "테스트항목", "시험 항목", "검증 항목", "점검 항목"
    ],
    "테스트 절차": [
        "테스트절차", "시험 절차", "검증 절차"
    ],
    "결함 내역": [
        "결함내역", "결함 목록", "오류 내역", "버그 내역"
    ],
    "조치 결과": [
        "조치결과", "결함 조치 결과", "수정 결과", "개선 결과"
    ],
    "검수 결과": [
        "검수결과", "검사 결과", "확인 결과"
    ],
    "검수 확인": [
        "검수확인", "검수 확인서", "검사 확인", "확인서"
    ],
}


@dataclass
class DeliverableAIReviewIssue:
    issue_type: str
    message: str
    title: str = ""
    sample: str = ""


@dataclass
class DeliverableAIReviewReport:
    passed: bool
    review_status: str
    document_type: str
    content_similarity: float
    expected_section_count: int
    parsed_section_count: int
    issues: List[DeliverableAIReviewIssue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class YongukDeliverableSubtitleAIReviewAgent:
    """
    Workit 산출물 소제목 AI 검수 에이전트.

    검수 기능:
    1. 산출물 원본 전체와 파싱 결과 전체 유사도 검수
    2. 기대 소제목 누락 검수
    3. 소제목 순서 검수
    4. 빈 소제목 본문 검수
    5. 한 소제목 안에 다른 소제목이 섞였는지 검수
    6. 원본 문단이 파싱 결과에서 누락되었는지 검수
    7. 숫자, 날짜, 금액, 퍼센트 등 중요 토큰 변경 검수
    """

    def __init__(
        self,
        expected_titles: Optional[List[str]] = None,
        document_type: Optional[str] = None,
        overall_similarity_threshold: float = PASS_SIMILARITY_THRESHOLD,
        block_similarity_threshold: float = 0.90,
    ):
        self.document_type = document_type or DEFAULT_DELIVERABLE_TYPE
        self.expected_titles = expected_titles or DELIVERABLE_TYPE_SECTION_TITLES.get(self.document_type, DEFAULT_DELIVERABLE_SECTION_TITLES)
        self.overall_similarity_threshold = overall_similarity_threshold
        self.block_similarity_threshold = block_similarity_threshold

    def review(
        self,
        original_text: str,
        parsed_sections: Any,
    ) -> DeliverableAIReviewReport:
        sections = self._normalize_parsed_sections(parsed_sections)
        parsed_text = self._join_sections(sections)

        content_similarity = self._similarity(original_text, parsed_text)

        issues: List[DeliverableAIReviewIssue] = []

        if not sections:
            issues.append(DeliverableAIReviewIssue(
                issue_type="empty_parsed_sections",
                message="산출물 소제목 파싱 결과가 비어 있습니다.",
            ))

        if content_similarity < self.overall_similarity_threshold:
            issues.append(DeliverableAIReviewIssue(
                issue_type="low_overall_similarity",
                message=f"산출물 원본과 파싱 결과의 전체 유사도가 낮습니다. similarity={content_similarity:.4f}",
            ))

        issues.extend(self._check_missing_sections(sections))
        issues.extend(self._check_section_order(original_text, sections))
        issues.extend(self._check_empty_sections(sections))
        issues.extend(self._check_overmerged_sections(sections))

        if content_similarity < self.overall_similarity_threshold:
            issues.extend(self._check_missing_original_blocks(original_text, parsed_text))
            issues.extend(self._check_added_parsed_blocks(original_text, parsed_text))

        token_issue = self._check_important_tokens_changed(
            original_text=original_text,
            parsed_text=parsed_text,
        )
        if token_issue:
            issues.append(token_issue)

        review_status = self._decide_review_status(content_similarity, issues)

        return DeliverableAIReviewReport(
            passed=review_status == "PASS",
            review_status=review_status,
            document_type=self.document_type,
            content_similarity=round(content_similarity, 4),
            expected_section_count=len(self.expected_titles),
            parsed_section_count=len(sections),
            issues=issues,
        )

    def _decide_review_status(
        self,
        content_similarity: float,
        issues: List[DeliverableAIReviewIssue],
    ) -> str:
        if content_similarity < 0.95:
            return "FAIL"

        critical_issue_types = {
            "empty_parsed_sections",
            "low_overall_similarity",
        }

        has_critical_issue = any(
            issue.issue_type in critical_issue_types
            for issue in issues
        )

        if has_critical_issue:
            return "FAIL"

        if content_similarity >= 0.98:
            return "PASS"

        return "WARN"
    def _normalize_parsed_sections(self, parsed_sections: Any) -> List[Dict[str, str]]:
        if not parsed_sections:
            return []

        if isinstance(parsed_sections, str):
            return [{
                "title": "",
                "content": parsed_sections.strip(),
            }]

        if isinstance(parsed_sections, dict):
            parsed_sections = [
                {"title": title, "content": content}
                for title, content in parsed_sections.items()
            ]

        sections = []

        for idx, item in enumerate(parsed_sections, start=1):
            if isinstance(item, str):
                sections.append({
                    "title": f"섹션{idx}",
                    "content": item.strip(),
                })
                continue

            if not isinstance(item, dict):
                continue

            title = (
                item.get("title")
                or item.get("section_title")
                or item.get("heading")
                or item.get("name")
                or item.get("section")
                or f"섹션{idx}"
            )

            content = (
                item.get("content")
                or item.get("text")
                or item.get("body")
                or item.get("section_text")
                or ""
            )

            sections.append({
                "title": str(title).strip(),
                "content": str(content).strip(),
            })

        return sections

    def _join_sections(self, sections: List[Dict[str, str]]) -> str:
        joined = []

        for section in sections:
            title = section.get("title", "").strip()
            content = section.get("content", "").strip()

            if title and content:
                joined.append(f"{title}\n{content}")
            elif content:
                joined.append(content)
            elif title:
                joined.append(title)

        return "\n".join(joined)

    def _check_missing_sections(
        self,
        sections: List[Dict[str, str]],
    ) -> List[DeliverableAIReviewIssue]:
        parsed_titles = {
            self._normalize_title(section.get("title", ""))
            for section in sections
        }

        issues = []

        for title in self.expected_titles:
            if self._is_expected_title_covered(title, parsed_titles):
                continue

            issues.append(DeliverableAIReviewIssue(
                issue_type="missing_section",
                title=title,
                message=f"기대 소제목 '{title}'이 파싱 결과에 없습니다.",
            ))

        return issues

    def _is_expected_title_covered(
            self,
            title: str,
            parsed_titles: set[str],
    ) -> bool:
        title_norm = self._normalize_title(title)

        if title_norm in parsed_titles:
            return True

        aliases = DELIVERABLE_TITLE_ALIASES.get(title, [])
        alias_norms = [self._normalize_title(alias) for alias in aliases]

        if any(alias_norm in parsed_titles for alias_norm in alias_norms):
            return True

        child_titles = DELIVERABLE_PARENT_TITLE_CHILDREN.get(self.document_type, {}).get(title, [])
        child_norms = [self._normalize_title(child) for child in child_titles]

        if child_norms and any(child_norm in parsed_titles for child_norm in child_norms):
            return True

        return False

    def _check_section_order(
        self,
        original_text: str,
        sections: List[Dict[str, str]],
    ) -> List[DeliverableAIReviewIssue]:
        original_positions = self._find_title_positions(original_text)

        expected_order = [
            self._normalize_title(title)
            for title in self.expected_titles
            if original_positions.get(title, -1) >= 0
        ]

        parsed_order = [
            self._normalize_title(section.get("title", ""))
            for section in sections
        ]

        parsed_order = [
            title for title in parsed_order
            if title in expected_order
        ]
        parsed_order = self._dedupe_preserve_order(parsed_order)

        expected_order = [
            title for title in expected_order
            if title in parsed_order
        ]

        if parsed_order and expected_order and parsed_order != expected_order:
            return [DeliverableAIReviewIssue(
                issue_type="section_order_mismatch",
                message="산출물 소제목 순서가 원본과 다릅니다.",
                sample=f"expected={expected_order}, parsed={parsed_order}",
            )]

        return []

    def _check_empty_sections(
        self,
        sections: List[Dict[str, str]],
    ) -> List[DeliverableAIReviewIssue]:
        issues = []

        for section in sections:
            title = section.get("title", "")
            content = section.get("content", "")

            if title and not content.strip():
                issues.append(DeliverableAIReviewIssue(
                    issue_type="empty_section",
                    title=title,
                    message=f"소제목 '{title}'은 존재하지만 내용이 비어 있습니다.",
                ))

        return issues

    def _check_overmerged_sections(
        self,
        sections: List[Dict[str, str]],
    ) -> List[DeliverableAIReviewIssue]:
        issues = []
        parent_titles = set(DELIVERABLE_PARENT_TITLE_CHILDREN.get(self.document_type, {}))

        for section in sections:
            current_title = section.get("title", "")
            current_norm = self._normalize_title(current_title)

            if self._is_front_matter_section(current_title):
                continue

            content_lines = self._content_lines_without_own_title(section)

            for expected_title in self.expected_titles:
                if expected_title in parent_titles:
                    continue

                expected_norm = self._normalize_title(expected_title)

                if not expected_norm or expected_norm == current_norm:
                    continue

                if self._contains_heading_line(content_lines, expected_title):
                    issues.append(DeliverableAIReviewIssue(
                        issue_type="overmerged_section",
                        title=current_title,
                        message=(
                            f"'{current_title}' 내용 안에 다른 소제목 '{expected_title}'이 제목 라인처럼 포함되어 있습니다. "
                            "소제목 분리 경계가 잘못되었을 가능성이 있습니다."
                        ),
                    ))
                    break

        return issues

    def _dedupe_preserve_order(self, items: List[str]) -> List[str]:
        result = []
        seen = set()

        for item in items:
            if item and item not in seen:
                result.append(item)
                seen.add(item)

        return result

    def _is_front_matter_section(self, title: str) -> bool:
        title_norm = self._normalize_title(title)
        return title_norm in {"서문", "목차"}

    def _content_lines_without_own_title(self, section: Dict[str, str]) -> List[str]:
        current_norm = self._normalize_title(section.get("title", ""))
        lines = []

        for raw_line in section.get("content", "").splitlines():
            line = raw_line.strip()
            if not line:
                continue

            if self._normalize_title(line) == current_norm:
                continue

            lines.append(line)

        return lines

    def _contains_heading_line(self, lines: List[str], expected_title: str) -> bool:
        expected_norm = self._normalize_title(expected_title)

        for line in lines:
            line_norm = self._normalize_title(line)
            compact_len = len(self._normalize_compare_text(line))

            if line_norm == expected_norm:
                return True

            if line_norm.startswith(expected_norm) and compact_len <= len(expected_norm) + 15:
                return True

        return False

    def _check_missing_original_blocks(
        self,
        original_text: str,
        parsed_text: str,
        max_samples: int = 5,
    ) -> List[DeliverableAIReviewIssue]:
        issues = []
        parsed_blocks = self._split_blocks(parsed_text)

        for original_block in self._split_blocks(original_text):
            if len(self._normalize_compare_text(original_block)) < 30:
                continue

            best_similarity = self._best_similarity(original_block, parsed_blocks)

            if best_similarity < self.block_similarity_threshold:
                issues.append(DeliverableAIReviewIssue(
                    issue_type="missing_original_content",
                    message="산출물 원본 내용 중 파싱 결과에서 누락되었거나 크게 달라진 부분이 있습니다.",
                    sample=original_block[:250],
                ))

            if len(issues) >= max_samples:
                break

        return issues

    def _check_added_parsed_blocks(
        self,
        original_text: str,
        parsed_text: str,
        max_samples: int = 5,
    ) -> List[DeliverableAIReviewIssue]:
        issues = []
        original_blocks = self._split_blocks(original_text)

        for parsed_block in self._split_blocks(parsed_text):
            if len(self._normalize_compare_text(parsed_block)) < 30:
                continue

            best_similarity = self._best_similarity(parsed_block, original_blocks)

            if best_similarity < self.block_similarity_threshold:
                issues.append(DeliverableAIReviewIssue(
                    issue_type="added_parsed_content",
                    message="파싱 결과에 산출물 원본에서 찾기 어려운 내용이 있습니다.",
                    sample=parsed_block[:250],
                ))

            if len(issues) >= max_samples:
                break

        return issues

    def _check_important_tokens_changed(
            self,
            original_text: str,
            parsed_text: str,
    ) -> Optional[DeliverableAIReviewIssue]:
        original_tokens = self._extract_important_tokens(original_text)
        parsed_tokens = self._extract_important_tokens(parsed_text)

        original_counter = Counter(original_tokens)
        parsed_counter = Counter(parsed_tokens)

        if original_counter == parsed_counter:
            return None

        missing_counter = original_counter - parsed_counter  # 원본에는 있는데 파싱엔 부족한 토큰
        added_counter = parsed_counter - original_counter  # 파싱에는 있는데 원본엔 없거나 더 많은 토큰

        missing_tokens = list(missing_counter.elements())
        added_tokens = list(added_counter.elements())

        sample_parts = []
        if missing_tokens:
            sample_parts.append(f"누락 토큰 일부: {missing_tokens[:20]}")
        if added_tokens:
            sample_parts.append(f"추가 토큰 일부: {added_tokens[:20]}")

        return DeliverableAIReviewIssue(
            issue_type="important_token_changed",
            message="원본과 파싱 결과의 숫자/날짜/금액/퍼센트 등 중요 토큰 구성이 다릅니다.",
            sample=" / ".join(sample_parts) if sample_parts else "중요 토큰 차이 발생",
        )

        return None

    def _find_title_positions(self, original_text: str) -> Dict[str, int]:
        original_norm = self._normalize_compare_text(original_text)
        positions = {}

        for title in self.expected_titles:
            title_norm = self._normalize_compare_text(title)
            positions[title] = original_norm.find(title_norm)

        return positions

    def _split_blocks(self, text: str) -> List[str]:
        text = self._clean_text(text)

        blocks = re.split(r"\n\s*\n", text)

        if len(blocks) <= 1:
            blocks = re.split(r"(?:[.!?。！？]\s+|다\.\s+|함\.\s+|음\.\s+)", text)

        return [
            block.strip()
            for block in blocks
            if block.strip()
        ]

    def _best_similarity(self, target: str, candidates: List[str]) -> float:
        target_norm = self._normalize_compare_text(target)
        best = 0.0

        for candidate in candidates:
            candidate_norm = self._normalize_compare_text(candidate)
            score = SequenceMatcher(None, target_norm, candidate_norm).ratio()
            best = max(best, score)

        return best

    def _extract_important_tokens(self, text: str) -> List[str]:
        patterns = [
            r"\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일",
            r"\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2}",
            r"\d+\s*개월",
            r"\d+\s*년",
            r"\d{1,3}(?:,\d{3})+\s*원",
            r"\d+\s*만원",
            r"\d+\s*억원",
            r"\d+\s*원",
            r"\d+(?:\.\d+)?\s*%",
        ]

        tokens = []

        for pattern in patterns:
            tokens.extend(re.findall(pattern, text or ""))

        return [
            self._normalize_compare_text(token)
            for token in tokens
            if self._normalize_compare_text(token)
        ]

    def _similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(
            None,
            self._normalize_compare_text(a),
            self._normalize_compare_text(b),
        ).ratio()

    def _normalize_title(self, title: str) -> str:
        title = title or ""

        # 제목 앞 번호 제거: 1. / 1) / Ⅰ. / ① / 가. 등
        title = re.sub(r"^\s*\d{1,3}[.)]\s*", "", title)
        title = re.sub(r"^\s*[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+[.)]?\s*", "", title)
        title = re.sub(r"^\s*[①②③④⑤⑥⑦⑧⑨⑩]\s*", "", title)
        title = re.sub(r"^\s*[가-하][.)]\s*", "", title)

        title = re.sub(r"\s+", "", title)
        return title.strip()

    def _normalize_compare_text(self, text: str) -> str:
        text = text or ""
        text = text.replace("\u00a0", " ")
        text = re.sub(r"\s+", "", text)
        text = re.sub(r"[()（）\[\]【】ㆍ·.,:;]", "", text)
        return text.strip()

    def _clean_text(self, text: str) -> str:
        text = text or ""
        text = text.replace("\u00a0", " ")

        # 목차 제목 제거
        text = re.sub(r"^\s*목\s*차\s*$", "", text, flags=re.MULTILINE)

        # 페이지 번호 제거: 1 / - 1 - / Page 1 등
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*-\s*\d+\s*-\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*Page\s*\d+\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE)

        # 안내성 문구 일부 제거
        text = re.sub(r"※\s*세부내용은.*?참조", "", text)

        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def review_deliverable_subtitle_ai_parsing(
    original_text: str,
    parsed_sections: Any,
    expected_titles: Optional[List[str]] = None,
    document_type: Optional[str] = None,
) -> Dict[str, Any]:
    agent = YongukDeliverableSubtitleAIReviewAgent(
        expected_titles=expected_titles,
        document_type=document_type,
    )
    return agent.review(
        original_text=original_text,
        parsed_sections=parsed_sections,
    ).to_dict()


def load_pdf_text(path: Path) -> str:
    if pdfplumber is None:
        raise ImportError("pdfplumber가 설치되어 있지 않습니다. pip install pdfplumber 실행이 필요합니다.")

    texts = []

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                texts.append(page_text.strip())

    return "\n\n".join(texts)


def load_json_sections(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_json_sections(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def print_deliverable_review_report(report: Dict[str, Any], source_name: str | None = None) -> None:
    issues = report.get("issues", [])
    review_status = report.get("review_status", "PASS" if report["passed"] else "FAIL")
    document_type = report.get("document_type", "unknown")
    content_similarity = report.get("content_similarity", 0)
    parsed_section_count = report.get("parsed_section_count", 0)
    elapsed_seconds = report.get("elapsed_seconds")

    fix_issue_types = {
        "empty_parsed_sections",
        "low_overall_similarity",
        "missing_original_content",
        "added_parsed_content",
    }

    reference_issue_types = {
        "missing_section",
        "section_order_mismatch",
        "empty_section",
        "overmerged_section",
        "important_token_changed",
    }

    fix_issues = []
    reference_issues = []

    for issue in issues:
        issue_type = issue.get("issue_type", "")

        if issue_type in fix_issue_types:
            fix_issues.append(issue)
        elif issue_type in reference_issue_types:
            reference_issues.append(issue)
        else:
            reference_issues.append(issue)

    if review_status == "PASS":
        result_line = "원본 대비 파싱 품질이 양호하여 추가 수정 없이 사용 가능한 상태입니다."
    elif review_status == "WARN":
        result_line = "전반적인 파싱 품질은 양호하지만 일부 구조/토큰 검토가 필요합니다."
    else:
        result_line = "원본과 파싱 결과 간 차이가 있어 수정 후 재검토가 필요합니다."

    summaries = []

    if content_similarity >= PASS_SIMILARITY_THRESHOLD:
        summaries.append("원본 내용이 파싱 결과에 거의 완전하게 보존됨")
    elif content_similarity >= WARN_SIMILARITY_THRESHOLD:
        summaries.append("원본 내용은 대부분 보존됨")
    else:
        summaries.append("원본과 파싱 결과의 내용 차이가 큼")

    if reference_issues:
        summaries.append("일부 소제목 분리 기준 또는 중요 토큰 확인 필요")
    else:
        summaries.append("소제목 구조와 중요 토큰 검수 결과가 전반적으로 안정적임")

    if any(issue.get("issue_type") == "important_token_changed" for issue in issues):
        summaries.append("중요 숫자/날짜/금액/퍼센트 정보 확인 필요")
    else:
        summaries.append("중요 숫자 정보 변경 없음")

    recommendations = []

    missing_titles = [
        issue.get("title")
        for issue in issues
        if issue.get("issue_type") == "missing_section" and issue.get("title")
    ]

    for title in missing_titles[:3]:
        recommendations.append(f"'{title}' 소제목 alias 추가 또는 기준 포함 여부 확인 필요")

    if any(issue.get("issue_type") == "overmerged_section" for issue in issues):
        recommendations.append("목차 영역 또는 소제목 경계가 과병합으로 잡혔는지 확인 필요")

    if any(issue.get("issue_type") == "important_token_changed" for issue in issues):
        recommendations.append("중요 숫자/날짜/금액/퍼센트 토큰 차이 확인 필요")

    if content_similarity < PASS_SIMILARITY_THRESHOLD:
        recommendations.append("목차 영역, 페이지 번호, 양식 안내 문구 등 비교 제외 대상 정제 권장")

    if not recommendations:
        recommendations.append("추가 수정 필요 없음")

    print()
    print("========== AI 검수 에이전트 ==========")

    if source_name:
        print(f"[검수 대상] {source_name}")

    print(f"[최종 판정] {review_status}")
    print(f"[문서 유형] {document_type}")
    print(f"[전체 유사도] {content_similarity}")

    print()
    print("[검수 결과]")
    print(f"- {result_line}")

    print()
    print("[검수 메타]")
    print(f"- 파싱 섹션 수: {parsed_section_count}개")
    print(f"- 전체 이슈 수: {len(issues)}개")
    print(f"- 수정 필요 이슈 수: {len(fix_issues)}개")
    print(f"- 참고 이슈 수: {len(reference_issues)}개")

    print()
    print("[판정 기준]")
    print(f"- PASS: 유사도 {PASS_SIMILARITY_THRESHOLD} 이상, 치명 이슈 없음")
    print(f"- WARN: 유사도 {WARN_SIMILARITY_THRESHOLD} 이상 또는 구조/토큰 확인 필요")
    print(f"- FAIL: 유사도 {WARN_SIMILARITY_THRESHOLD} 미만 또는 원본 누락/파싱 실패")

    print()
    print("[핵심 요약]")
    for summary in summaries:
        print(f"- {summary}")

    print()
    print("[수정 권장]")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")

    def _print_issue_list(title: str, issue_list: List[Dict[str, Any]]) -> None:
        print()
        print(f"[{title}]")
        if not issue_list:
            print("- 없음")
            return

        for issue in issue_list:
            issue_type = issue.get("issue_type", "-")
            issue_title = issue.get("title") or "-"
            message = issue.get("message", "")
            sample = issue.get("sample", "")

            print(f"- {issue_type} / {issue_title}: {message}")
            if sample:
                print(f"  예시: {sample}")

    _print_issue_list("수정 필요", fix_issues)
    _print_issue_list("참고 이슈", reference_issues)

    if elapsed_seconds is not None:
        print()
        print("[실행 정보]")
        print(f"- 검수 소요시간: {elapsed_seconds:.2f}초")


def detect_deliverable_document_type(filename: str, text: str) -> str:
    source = f"{filename}\n{text[:3000]}"

    if any(keyword in source for keyword in ["테스트계획서", "테스트 결과", "테스트결과", "검수확인서", "검수 확인"]):
        return "test_verification"

    if any(keyword in source for keyword in ["최종완료", "최종결과", "사업추진결과", "완료보고", "결과보고"]):
        return "result_completion"

    if any(keyword in source for keyword in ["제안요청서", "과업수행계획서", "사업수행계획서", "요구사항", "제안요청"]):
        return "planning_requirement"

    return DEFAULT_DELIVERABLE_TYPE


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    original_path = project_root / "data" / "deliverable" / ""
    parsed_path = project_root / "data" / "structured" / ""

    if not original_path.exists():
        print(f"[ERROR] 산출물 원본 파일이 없습니다: {original_path}")
        return

    if not parsed_path.exists():
        print(f"[ERROR] 산출물 파싱 JSON 파일이 없습니다: {parsed_path}")
        return

    original_text = load_pdf_text(original_path)
    parsed_sections = load_json_sections(parsed_path)
    document_type = detect_deliverable_document_type(original_path.name, original_text)

    start_time = time.perf_counter()

    report = review_deliverable_subtitle_ai_parsing(
        original_text=original_text,
        parsed_sections=parsed_sections,
        document_type=document_type,
    )

    report["elapsed_seconds"] = time.perf_counter() - start_time

    print_deliverable_review_report(report, source_name=original_path.name)


if __name__ == "__main__":
    main()


