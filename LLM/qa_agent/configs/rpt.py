from qa_agent.section_spec import SectionSpec

# 사업추진결과보고서 (RPT) 소제목 체계
SECTIONS = [
    SectionSpec("RPT-01-01", "제1절 개요", "제1장 사업개요", []),
    SectionSpec("RPT-01-02", "제2절 사업의 배경 및 목적", "제1장 사업개요", []),
    SectionSpec("RPT-01-03", "제3절 사업추진체계", "제1장 사업개요", []),
    SectionSpec("RPT-01-04", "제4절 추진경과", "제1장 사업개요", []),

    SectionSpec("RPT-02-01", "제1절 적용방법론", "제2장 사업내용(개발사업)", []),
    SectionSpec("RPT-02-02", "제2절 개발내용", "제2장 사업내용(개발사업)", []),
    SectionSpec("RPT-02-03-01", "제3절 시스템 구성도", "제2장 사업내용(개발사업)", []),
    SectionSpec("RPT-02-04-01", "제4절 표준화 적용결과", "제2장 사업내용(개발사업)", []),
    SectionSpec("RPT-02-05", "제5절 보안 부문", "제2장 사업내용(개발사업)", []),
    SectionSpec("RPT-02-06", "제6절 법제도 정비실적", "제2장 사업내용(개발사업)", []),

    SectionSpec("RPT-03-01", "제1절 운영계획", "제3장 운영계획 및 발전방향", []),
    SectionSpec("RPT-03-02", "제2절 발전방향", "제3장 운영계획 및 발전방향", []),
]
