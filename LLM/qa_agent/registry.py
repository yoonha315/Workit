from __future__ import annotations

from typing import List

from qa_agent.section_spec import SectionSpec
from qa_agent.configs import rfp, pep, rpt

# 새 문서 유형이 생기면 configs/ 밑에 파일 하나 추가하고 여기에 등록만 하면 됨.
REGISTRY = {
    "rfp": rfp.SECTIONS,
    "pep": pep.SECTIONS,
    "rpt": rpt.SECTIONS,
}


def get_sections(document_type: str) -> List[SectionSpec]:
    if document_type not in REGISTRY:
        raise ValueError(
            f"알 수 없는 document_type: '{document_type}' "
            f"(사용 가능: {list(REGISTRY.keys())})"
        )
    return REGISTRY[document_type]
