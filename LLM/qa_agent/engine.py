from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from qa_agent.section_spec import SectionSpec
from qa_agent.text_utils import (
    normalize_compare_text,
    clean_text,
    similarity,
    normalize_with_map,
    containment_ratio,
    split_paragraphs,
)


# ---------------------------------------------------------------------------
# 판정 기준값
# ---------------------------------------------------------------------------

PASS_SIMILARITY_THRESHOLD = 0.98
WARN_SIMILARITY_THRESHOLD = 0.95  # 현재는 판정에 안 쓰고, 참고용 유사도 계산 컷오프로만 남겨둠
SECTION_CONTENT_THRESHOLD = 0.75   # 이 밑으로 떨어지면 섹션 내용이 원본과 안 맞는다고 판단
MIN_CONTENT_LENGTH = 5              # 정규화 후 이보다 짧으면 사실상 빈 섹션으로 간주
PARAGRAPH_MIN_LENGTH = 20            # 이보다 짧은 문단은 노이즈 가능성이 높아 문단 단위 검사에서 제외
PARAGRAPH_CONTAINMENT_THRESHOLD = 0.7  # 문단이 어떤 content 안에 이 비율 이상 들어있어야 '거기 있다'고 인정

# 반려 코멘트를 반드시 띄우고, 자동 진행을 막아야 하는 이슈들
BLOCKING_ISSUE_TYPES = {
    "missing_section",           # 필수 소제목 자체가 없음
    "empty_section",             # 소제목은 있는데 내용이 텅 빔
    "section_content_mismatch",  # 내용이 원본과 안 맞음 (다른 곳으로 넘어갔거나 유실)
    "content_misplaced",         # 내용이 다른 소제목 자리에 잘못 들어감
    "paragraph_misplaced",       # 섹션 전체는 대체로 맞는데, 그 안의 문단 하나가 다른 섹션으로 넘어감
    "paragraph_missing",         # 섹션 전체는 대체로 맞는데, 그 안의 문단 하나가 어디서도 안 보임
}

# 참고용으로만 보여주고 진행은 막지 않는 이슈들
INFO_ISSUE_TYPES = {
    "unrecognized_section",   # 기준 목록에 없는 소제목이 파싱 결과에 있음 (신규 항목 가능성)
    "section_order_mismatch",  # 순서만 다르고 내용 자체는 문제없음
}


# ---------------------------------------------------------------------------
# 결과 데이터 구조
# ---------------------------------------------------------------------------

@dataclass
class SectionIssue:
    issue_type: str
    code: str = ""
    title: str = ""
    message: str = ""
    sample: str = ""
    severity: str = "info"   # "blocking" (FAIL 유발) 또는 "info" (참고용)


@dataclass
class SectionReviewReport:
    passed: bool
    review_status: str          # PASS / WARN / FAIL
    can_auto_proceed: bool      # 반려 코멘트 없이 바로 문서검토로 진행 가능한지
    document_type: str
    content_similarity: float
    expected_section_count: int
    matched_section_count: int
    issues: List[SectionIssue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 엔진
# ---------------------------------------------------------------------------

class SectionMappingReviewAgent:
    """
    소제목 코드 체계(SectionSpec 목록)를 기준으로,
    '원본 텍스트'와 'AI가 파싱한 소제목-내용 key-value 결과'가
    소제목 단위로 정확하게 매핑됐는지 검수한다.

    검수 항목:
      1. 기대 소제목 누락 검수 (+ alias 매칭)
      2. 빈 소제목 검수
      3. 기준 목록에 없는 소제목이 파싱 결과에 있는지 검수
      4. 소제목 순서 검수
      5. 섹션별 내용 매핑 검수
         - 원본에서 그 소제목의 정확한 구간을 잘라내고
         - 파싱 결과의 같은 소제목 내용과 비교해서
         - 내용 유실/변형(section_content_mismatch) 또는
           다른 소제목으로의 오배치(content_misplaced)를 구분해서 잡아냄
      6. 문단 단위 검수
         - 5번은 섹션 전체를 한 덩어리로 보기 때문에, 문단 여러 개 중 하나만
           다른 섹션으로 새어나간 경우 전체 유사도가 threshold를 넘겨서 놓칠 수 있음
         - 원본을 문단 단위로 쪼개서 각 문단이 자기 섹션 content 안에 있는지,
           없으면 다른 섹션으로 넘어간 건지(paragraph_misplaced) 아예 안 보이는
           건지(paragraph_missing)를 문단 단위로 재확인
      7. 문서 전체 유사도 (참고 지표)

    입력(parsed_sections)은 두 형태를 다 지원한다:
      - 구버전: {"파싱된 소제목 텍스트": "내용", ...} 형태의 flat dict
      - 신규: [{"section_id": "pep_03_01", "section_title": "...", "content": "...", ...}, ...]
        형태의 레코드 리스트 (section_id로 code를 직접 매칭하므로 더 안정적)
    """

    def __init__(
        self,
        sections: List[SectionSpec],
        document_type: str,
        overall_pass_threshold: float = PASS_SIMILARITY_THRESHOLD,
        overall_warn_threshold: float = WARN_SIMILARITY_THRESHOLD,
        section_content_threshold: float = SECTION_CONTENT_THRESHOLD,
    ):
        self.sections = sections
        self.document_type = document_type
        self.overall_pass_threshold = overall_pass_threshold
        self.overall_warn_threshold = overall_warn_threshold
        self.section_content_threshold = section_content_threshold

    # ------------------------------------------------------------------
    # 진입점
    # ------------------------------------------------------------------
    def review(
        self,
        original_text: str,
        parsed_sections: Any,
    ) -> SectionReviewReport:
        original_clean = clean_text(original_text)
        parsed_sections = parsed_sections or {}

        matched, unrecognized_keys, parsed_order_codes = self._parse_input(parsed_sections)

        issues: List[SectionIssue] = []

        # 1. 소제목 누락 검수
        for spec in self.sections:
            if spec.code not in matched:
                issues.append(SectionIssue(
                    issue_type="missing_section",
                    code=spec.code,
                    title=spec.title,
                    message=f"기대 소제목 '{spec.title}'({spec.code})이 파싱 결과에서 확인되지 않습니다.",
                ))

        # 2. 빈 소제목 검수
        empty_codes: set = set()
        for code, info in matched.items():
            content = (info.get("content") or "").strip()
            if len(normalize_compare_text(content)) < MIN_CONTENT_LENGTH:
                empty_codes.add(code)
                spec = self._spec_by_code(code)
                issues.append(SectionIssue(
                    issue_type="empty_section",
                    code=code,
                    title=spec.title if spec else "",
                    message=f"'{info.get('parsed_title')}'({code}) 소제목은 있지만 내용이 비어 있거나 지나치게 짧습니다.",
                ))

        # 3. 기준 목록에 없는 소제목 검수
        for key in unrecognized_keys:
            issues.append(SectionIssue(
                issue_type="unrecognized_section",
                title=key,
                message=f"파싱 결과의 '{key}' 항목은 기준 소제목 목록에 없습니다. 신규 항목이거나 소제목 분리가 잘못됐을 수 있습니다.",
            ))

        # 원본에서 각 소제목의 위치를 순차적으로 탐색.
        # matched 여부와 무관하게 config에 정의된 모든 소제목을 대상으로 찾아야
        # (파싱 결과에서 누락된 소제목이 있어도) 옆 섹션의 구간 경계가 정확하게 잡힌다.
        all_codes = [spec.code for spec in self.sections]
        positions, original_norm, index_map = self._find_section_positions(original_clean, all_codes)

        # 4. 순서 검수
        issues.extend(self._check_order(positions, parsed_order_codes))

        # 5. 섹션별 내용 매핑 검수 (이미 empty_section으로 잡힌 섹션은 제외 - 중복/약한 오탐 방지)
        section_level_issues = self._check_section_content_mapping(
            original_norm, positions, matched, empty_codes
        )
        issues.extend(section_level_issues)

        # 6. 문단 단위 검수: 섹션 전체로 보면 그럴듯해서 5번 검사를 통과했지만,
        #    그 안의 문단 하나가 실제로는 다른 섹션으로 넘어갔거나 사라진 경우를 잡아낸다.
        #    (섹션 전체가 이미 misplaced/mismatch로 잡힌 곳은 중복 노이즈라 건너뜀)
        already_flagged_codes = {
            issue.code for issue in section_level_issues
            if issue.issue_type in ("content_misplaced", "section_content_mismatch")
        }
        issues.extend(self._check_paragraph_level_content(
            original_clean, index_map, positions, matched, empty_codes | already_flagged_codes
        ))

        for issue in issues:
            issue.severity = "blocking" if issue.issue_type in BLOCKING_ISSUE_TYPES else "info"

        # 7. 문서 전체 유사도 (참고 지표 - 판정에는 blocking 이슈가 없을 때만 보조적으로 사용)
        joined_parsed_text = "\n".join((info.get("content") or "") for info in matched.values())
        content_similarity = similarity(original_text, joined_parsed_text)

        review_status, can_auto_proceed = self._decide_status(content_similarity, issues)

        return SectionReviewReport(
            passed=review_status == "PASS",
            review_status=review_status,
            can_auto_proceed=can_auto_proceed,
            document_type=self.document_type,
            content_similarity=round(content_similarity, 4),
            expected_section_count=len(self.sections),
            matched_section_count=len(matched),
            issues=issues,
        )

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------
    def _spec_by_code(self, code: str) -> Optional[SectionSpec]:
        for spec in self.sections:
            if spec.code == code:
                return spec
        return None

    def _parse_input(
        self, parsed_sections: Any
    ) -> Tuple[Dict[str, Dict[str, Any]], List[str], List[str]]:
        """
        파싱 결과 입력을 두 가지 형태 다 지원한다:
          1) 구버전: {"소제목 텍스트": "내용", ...} 형태의 flat dict
             -> title/alias 문자열 매칭으로 code를 찾는다.
          2) 신규: [{"section_id": "pep_03_01", "section_title": "...", "content": "...", ...}, ...]
             형태의 레코드 리스트
             -> section_id를 code로 직접 변환해서 매칭한다 (title 표현 차이에 영향 안 받아 더 안정적).

        반환값: (matched: code -> {parsed_title, content}, unrecognized_labels, parsed_order_codes)
        """
        if isinstance(parsed_sections, dict):
            return self._parse_flat_dict(parsed_sections)
        if isinstance(parsed_sections, list):
            return self._parse_record_list(parsed_sections)
        raise TypeError(
            "parsed_sections는 {'소제목': '내용'} 형태의 dict 또는 "
            "[{'section_id':.., 'section_title':.., 'content':..}, ...] 형태의 list여야 합니다. "
            f"입력 타입: {type(parsed_sections)}"
        )

    def _parse_flat_dict(
        self, parsed_sections: Dict[str, str]
    ) -> Tuple[Dict[str, Dict[str, Any]], List[str], List[str]]:
        """title/alias 문자열 매칭 (구버전 입력 형태)."""
        matched: Dict[str, Dict[str, Any]] = {}
        used_keys: set = set()

        for spec in self.sections:
            candidate_norms = {normalize_compare_text(c) for c in spec.title_candidates()}
            for key, content in parsed_sections.items():
                if key in used_keys:
                    continue
                if normalize_compare_text(key) in candidate_norms:
                    matched[spec.code] = {"parsed_title": key, "content": content}
                    used_keys.add(key)
                    break

        code_by_key = {info["parsed_title"]: code for code, info in matched.items()}
        parsed_order_codes = [code_by_key[k] for k in parsed_sections.keys() if k in code_by_key]
        unrecognized = [k for k in parsed_sections.keys() if k not in used_keys]

        return matched, unrecognized, parsed_order_codes

    def _parse_record_list(
        self, records: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Dict[str, Any]], List[str], List[str]]:
        """section_id 기반 매칭 (신규 입력 형태). title 텍스트 차이에 영향받지 않아 더 안정적."""
        matched: Dict[str, Dict[str, Any]] = {}
        parsed_order_codes: List[str] = []
        unrecognized: List[str] = []
        valid_codes = {spec.code for spec in self.sections}

        for record in records:
            if not isinstance(record, dict):
                continue

            section_id = str(record.get("section_id") or "").strip()
            code = self._section_id_to_code(section_id)
            title = record.get("section_title") or record.get("title") or ""
            content = record.get("content") or ""

            if code in valid_codes and code not in matched:
                matched[code] = {"parsed_title": title, "content": content, "section_id": section_id}
                parsed_order_codes.append(code)
            else:
                unrecognized.append(section_id or title or "(section_id 없음)")

        return matched, unrecognized, parsed_order_codes

    @staticmethod
    def _section_id_to_code(section_id: str) -> str:
        """
        'pep_03_01' -> 'PEP-03-01', 'pep_10' -> 'PEP-10' 형태로 변환.
        config의 code 표기(RFP-04-04-01 등)와 맞춘다.
        """
        if not section_id:
            return ""
        parts = section_id.strip().split("_")
        if len(parts) < 2:
            return section_id.upper()
        prefix = parts[0].upper()
        rest = "-".join(parts[1:])
        return f"{prefix}-{rest}"

    # 목차 탐지 기준값 (둘 다 만족해야 "목차가 있다"고 판단)
    _TOC_WINDOW_RATIO = 0.15   # 문서 앞부분 이 비율 이내에 몰려 있어야 목차 후보
    _TOC_MIN_CLUSTER_RATIO = 0.5  # 매칭된 소제목의 이 비율 이상이 그 구간에 몰려 있어야 함

    def _sequential_search(
        self, original_norm: str, ordered_specs: List[SectionSpec], start_from: int
    ) -> Dict[str, Tuple[int, str]]:
        """cursor 이후 '첫 매치'를 순서대로 찾아나간다 (짧고 흔한 단어가 본문 뒤쪽
        엉뚱한 곳에서 매치되는 걸 막기 위해, 무조건 마지막 매치를 쓰지는 않는다)."""
        positions: Dict[str, Tuple[int, str]] = {}
        cursor = start_from

        for spec in ordered_specs:
            found_idx = -1
            found_text = ""

            for candidate in spec.title_candidates():
                candidate_norm = normalize_compare_text(candidate)
                if not candidate_norm:
                    continue
                idx = original_norm.find(candidate_norm, cursor)
                if idx >= 0:
                    found_idx = idx
                    found_text = candidate
                    break

            if found_idx >= 0:
                positions[spec.code] = (found_idx, found_text)
                cursor = found_idx + len(normalize_compare_text(found_text))

        return positions

    def _find_section_positions(
        self, original_clean: str, codes: Any
    ) -> Tuple[Dict[str, Tuple[int, str]], str, List[int]]:
        """
        원본에서 각 소제목이 실제로 등장하는 위치를 찾는다.

        기본은 '직전에 찾은 위치 다음의 첫 매치'로 순서대로 찾아나가는 방식이다.
        다만 문서 앞부분에 목차가 있으면(대부분의 정형 문서가 그렇다) 목차에도
        소제목 문구가 본문과 똑같은 순서로 나열돼 있어서, 이 방식만으로는 검색
        커서가 목차 블록 안에서만 맴돌고 실제 본문(항상 목차보다 뒤에 있음)까지
        못 넘어가는 문제가 있다 — contracts/parsers.py의 parse_rfp()가 이미 겪었던
        문제와 동일하다.

        그래서 1차로 처음부터 순차 탐색을 해본 뒤, 매칭된 위치의 상당수가 문서
        앞부분 좁은 구간에 몰려 있으면(= 목차로 추정) 그 구간 바로 다음부터
        다시 순차 탐색해서 실제 본문 위치를 잡는다. 단순히 '마지막 위치 사용'으로
        바꾸지 않는 이유는, 성능·보안처럼 짧고 흔한 단어는 본문 뒤쪽 다른 섹션의
        설명 안에서도 다시 등장할 수 있어 마지막 매치가 오히려 엉뚱한 곳을 가리킬
        수 있기 때문이다.

        정규화된 텍스트(original_norm)와, 그 정규화 텍스트의 각 글자가 원본
        original_clean의 몇 번째 글자였는지 알려주는 index_map도 같이 반환한다.
        이 매핑이 있어야, 문단 단위 검사에서 줄바꿈이 살아있는 원본 그대로
        섹션 구간을 잘라낼 수 있다.
        """
        original_norm, index_map = normalize_with_map(original_clean)
        ordered_specs = [s for s in self.sections if s.code in codes]

        positions = self._sequential_search(original_norm, ordered_specs, start_from=0)

        doc_len = len(original_norm)
        toc_window = doc_len * self._TOC_WINDOW_RATIO
        clustered = [pos for pos, _ in positions.values() if pos <= toc_window]

        if positions and len(clustered) >= len(positions) * self._TOC_MIN_CLUSTER_RATIO:
            toc_end = max(clustered)
            retried = self._sequential_search(original_norm, ordered_specs, start_from=toc_end)
            if retried:
                positions = retried

        return positions, original_norm, index_map

    def _check_order(
        self,
        positions: Dict[str, Tuple[int, str]],
        parsed_order_codes: List[str],
    ) -> List[SectionIssue]:
        # 원본에 실제로 등장한 순서 (위치 기준 정렬)
        expected_order = sorted(positions.keys(), key=lambda c: positions[c][0])

        # 서로 공통으로 존재하는 code만 비교 대상으로 삼는다 (누락은 이미 별도 이슈로 처리됨)
        common = set(expected_order) & set(parsed_order_codes)
        expected_filtered = [c for c in expected_order if c in common]
        parsed_filtered = self._dedupe_preserve_order([c for c in parsed_order_codes if c in common])

        if expected_filtered and parsed_filtered and expected_filtered != parsed_filtered:
            return [SectionIssue(
                issue_type="section_order_mismatch",
                message="소제목 순서가 원본과 다릅니다.",
                sample=f"원본 순서={expected_filtered}, 파싱 순서={parsed_filtered}",
            )]

        return []

    def _check_section_content_mapping(
        self,
        original_norm: str,
        positions: Dict[str, Tuple[int, str]],
        matched: Dict[str, Dict[str, Any]],
        empty_codes: Optional[set] = None,
    ) -> List[SectionIssue]:
        empty_codes = empty_codes or set()
        issues: List[SectionIssue] = []

        # code -> 원본에서의 해당 구간(정규화 텍스트). 위치 순서대로 다음 소제목 직전까지 자름.
        ordered_codes = sorted(positions.keys(), key=lambda c: positions[c][0])
        slices: Dict[str, str] = {}

        for i, code in enumerate(ordered_codes):
            start = positions[code][0]
            end = positions[ordered_codes[i + 1]][0] if i + 1 < len(ordered_codes) else len(original_norm)
            raw_slice = original_norm[start:end]

            # 구간 맨 앞에 소제목 텍스트 자체가 붙어있는데, 파싱 결과 content엔
            # 보통 소제목이 안 들어있으므로(제목은 key, 내용은 value로 분리) 비교 전에 제거한다.
            # 안 그러면 짧은 섹션일수록 제목 길이만큼 유사도가 부당하게 깎인다.
            matched_title_norm = normalize_compare_text(positions[code][1])
            if matched_title_norm and raw_slice.startswith(matched_title_norm):
                raw_slice = raw_slice[len(matched_title_norm):]

            slices[code] = raw_slice

        for code, original_slice in slices.items():
            info = matched.get(code)
            spec = self._spec_by_code(code)

            if info is None:
                # 원본엔 있는데 파싱 결과에서 못 찾은 경우는 missing_section에서 이미 처리됨
                continue

            if code in empty_codes:
                # 이미 empty_section으로 잡힌 섹션은 여기서 다시 "어디로 잘못 들어갔나"를
                # 억지로 찾지 않는다 (약한 유사도 후보를 misplaced로 오탐할 수 있음)
                continue

            parsed_content = info.get("content") or ""
            own_similarity = similarity(original_slice, parsed_content)

            if own_similarity >= self.section_content_threshold:
                continue  # 내용이 잘 들어감

            # 자기 섹션과는 유사도가 낮은데, 혹시 다른 섹션 내용과 더 비슷한 건 아닌지 확인
            best_other_code = None
            best_other_score = own_similarity

            for other_code, other_info in matched.items():
                if other_code == code:
                    continue
                other_score = similarity(original_slice, other_info.get("content") or "")
                if other_score > best_other_score:
                    best_other_score = other_score
                    best_other_code = other_code

            title = spec.title if spec else code

            if best_other_code is not None:
                other_spec = self._spec_by_code(best_other_code)
                other_title = other_spec.title if other_spec else best_other_code
                issues.append(SectionIssue(
                    issue_type="content_misplaced",
                    code=code,
                    title=title,
                    message=(
                        f"'{title}'({code}) 원본 내용이 '{other_title}'({best_other_code}) 섹션에 "
                        f"잘못 들어간 것으로 보입니다. (자기 섹션 유사도 {own_similarity:.2f} vs "
                        f"'{best_other_code}' 유사도 {best_other_score:.2f})"
                    ),
                    sample=original_slice[:200],
                ))
            else:
                issues.append(SectionIssue(
                    issue_type="section_content_mismatch",
                    code=code,
                    title=title,
                    message=(
                        f"'{title}'({code}) 섹션의 원본 내용이 파싱 결과에 제대로 반영되지 않은 것으로 "
                        f"보입니다. (유사도 {own_similarity:.2f})"
                    ),
                    sample=original_slice[:200],
                ))

        return issues

    def _check_paragraph_level_content(
        self,
        original_clean: str,
        index_map: List[int],
        positions: Dict[str, Tuple[int, str]],
        matched: Dict[str, Dict[str, Any]],
        skip_codes: set,
    ) -> List[SectionIssue]:
        """
        섹션 전체 단위 비교(_check_section_content_mapping)는 '전체적으로 비슷한가'만
        보기 때문에, 섹션 안의 문단 여러 개 중 하나만 다른 섹션으로 새어나간 경우
        전체 유사도가 threshold를 넘겨버려서 못 잡을 수 있다. 이 함수는 그 틈을
        메우기 위해, 원본을 문단 단위로 쪼개서 문단 하나하나가 실제로 자기 섹션
        content 안에 들어있는지 확인한다.

        skip_codes에 있는 코드(이미 empty_section이거나 섹션 전체가 misplaced/mismatch로
        잡힌 경우)는 중복 이슈를 막기 위해 건너뛴다 - 이 검사는 '전체로 보면 괜찮아
        보이는데 그 안에 숨어있는 문제'를 잡기 위한 것이라서다.
        """
        issues: List[SectionIssue] = []
        ordered_codes = sorted(positions.keys(), key=lambda c: positions[c][0])

        # code -> 원본에서의 해당 구간 (줄바꿈이 살아있는 raw 텍스트, 문단 분리를 위해)
        raw_slices: Dict[str, str] = {}
        for i, code in enumerate(ordered_codes):
            norm_start = positions[code][0]
            norm_end = positions[ordered_codes[i + 1]][0] if i + 1 < len(ordered_codes) else len(index_map)
            raw_start = index_map[norm_start] if norm_start < len(index_map) else len(original_clean)
            raw_end = index_map[norm_end] if norm_end < len(index_map) else len(original_clean)
            raw_slice = original_clean[raw_start:raw_end]

            # 맨 앞 줄이 소제목 텍스트 자체인 경우가 많으므로 첫 줄을 제거해서 본문만 남긴다.
            title_norm = normalize_compare_text(positions[code][1])
            lines = raw_slice.split("\n", 1)
            if len(lines) > 1 and normalize_compare_text(lines[0]) == title_norm:
                raw_slice = lines[1]

            raw_slices[code] = raw_slice

        for code, raw_slice in raw_slices.items():
            if code in skip_codes or code not in matched:
                continue

            own_content = matched[code].get("content") or ""
            spec = self._spec_by_code(code)
            title = spec.title if spec else code

            for paragraph in split_paragraphs(raw_slice):
                if len(normalize_compare_text(paragraph)) < PARAGRAPH_MIN_LENGTH:
                    continue  # 너무 짧은 조각은 노이즈일 가능성이 높아 제외

                own_score = containment_ratio(paragraph, own_content)
                if own_score >= PARAGRAPH_CONTAINMENT_THRESHOLD:
                    continue  # 이 문단은 자기 섹션 안에 잘 들어있음

                best_other_code = None
                best_other_score = own_score
                for other_code, other_info in matched.items():
                    if other_code == code:
                        continue
                    other_score = containment_ratio(paragraph, other_info.get("content") or "")
                    if other_score > best_other_score:
                        best_other_score = other_score
                        best_other_code = other_code

                if best_other_code is not None and best_other_score >= PARAGRAPH_CONTAINMENT_THRESHOLD:
                    other_spec = self._spec_by_code(best_other_code)
                    other_title = other_spec.title if other_spec else best_other_code
                    issues.append(SectionIssue(
                        issue_type="paragraph_misplaced",
                        code=code,
                        title=title,
                        message=(
                            f"'{title}'({code}) 안의 문단 하나가 '{other_title}'({best_other_code}) 쪽으로 "
                            f"넘어간 것으로 보입니다. (자기 섹션 포함비율 {own_score:.2f} vs "
                            f"'{best_other_code}' 포함비율 {best_other_score:.2f})"
                        ),
                        sample=paragraph[:200],
                    ))
                else:
                    issues.append(SectionIssue(
                        issue_type="paragraph_missing",
                        code=code,
                        title=title,
                        message=(
                            f"'{title}'({code}) 안의 문단 하나가 파싱 결과 어디에서도 확인되지 않습니다. "
                            f"(자기 섹션 포함비율 {own_score:.2f})"
                        ),
                        sample=paragraph[:200],
                    ))

        return issues

    def _decide_status(
        self,
        content_similarity: float,
        issues: List[SectionIssue],
    ) -> Tuple[str, bool]:
        """
        PASS/FAIL 이분법: blocking 이슈(BLOCKING_ISSUE_TYPES)가 하나라도 있으면 FAIL,
        없으면 PASS. informational 이슈(unrecognized_section, section_order_mismatch)는
        issues 목록엔 그대로 남아서 참고용으로 보여지지만, 판정 자체에는 영향을 주지 않는다.

        content_similarity는 전체 유사도 참고용 지표로만 리포트에 남긴다.
        (원본 전체엔 목차/장 제목/소제목 텍스트가 포함되지만 파싱 결과엔 본문만 있어서,
        섹션이 전부 완벽히 매칭돼도 전체 유사도가 구조적으로 낮게 나올 수 있어
        판정 기준으로는 쓰지 않는다.)
        """
        has_blocking = any(issue.issue_type in BLOCKING_ISSUE_TYPES for issue in issues)

        status = "FAIL" if has_blocking else "PASS"
        can_auto_proceed = not has_blocking

        return status, can_auto_proceed

    @staticmethod
    def _dedupe_preserve_order(items: List[str]) -> List[str]:
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result


# ---------------------------------------------------------------------------
# 외부에서 부르기 편한 진입 함수
# ---------------------------------------------------------------------------

def review_section_mapping(
    original_text: str,
    parsed_sections: Any,
    document_type: str,
) -> Dict[str, Any]:
    """
    백엔드 어디서든 이 함수 하나만 import해서 쓰면 된다.
    원본 텍스트 + 파싱 결과(dict 또는 section_id 포함 레코드 리스트) + 문서유형
    문자열("rfp"/"pep"/"rpt")을 넣으면 바로 JSON 직렬화 가능한 dict 리포트가 나온다.
    """
    from qa_agent.registry import get_sections

    sections = get_sections(document_type)
    agent = SectionMappingReviewAgent(sections=sections, document_type=document_type)
    return agent.review(original_text=original_text, parsed_sections=parsed_sections).to_dict()
