from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SectionSpec:
    """
    문서 유형(RFP/PEP/RPT)마다 이름은 다르지만("장/절", "항목명/소제목" 등),
    구조적으로는 다 같다: 코드 + 상위 그룹 + 실제 매칭 대상 제목.

    code    : RFP-01-01, PEP-03-01, RPT-01-02 같은 고유 식별자
    title   : 실제로 원본/파싱 결과에서 찾아야 하는 소제목 텍스트 (정규 표현)
    group   : 상위 그룹(장/항목명). 순서/맥락 파악용이며 매칭에는 직접 안 씀. 없으면 None
    aliases : title과 같은 뜻으로 쓰일 수 있는 다른 표현들
    """
    code: str
    title: str
    group: Optional[str] = None
    aliases: List[str] = field(default_factory=list)

    def title_candidates(self) -> List[str]:
        """title 자신 + alias들을 합친, 매칭 시도할 후보 표현 목록."""
        return [self.title] + list(self.aliases)
