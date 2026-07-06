from qa_agent.section_spec import SectionSpec

# 사업수행계획서 (PEP) 소제목 체계
SECTIONS = [
    SectionSpec("PEP-01", "사업명", None, []),
    SectionSpec("PEP-02", "사업기간", None, []),

    SectionSpec("PEP-03-01", "추진배경", "사업목적", ["추진 배경"]),
    SectionSpec("PEP-03-02", "목적", "사업목적", ["사업 목적"]),

    SectionSpec("PEP-04-01", "개발대상업무", "사업범위", []),
    SectionSpec("PEP-04-02", "개발 및 운영환경", "사업범위", []),
    SectionSpec("PEP-04-03", "기타", "사업범위", []),

    SectionSpec("PEP-05-01", "총괄추진체계", "사업추진체계", []),
    SectionSpec("PEP-05-02", "사업자 추진체계", "사업추진체계", []),

    SectionSpec("PEP-06", "사업추진절차", None, []),
    SectionSpec("PEP-07", "산출물계획", None, []),
    SectionSpec("PEP-08", "일정계획", None, []),
    SectionSpec("PEP-09", "공정별 투입인력계획", None, []),
    SectionSpec("PEP-10", "보고계획", None, []),

    SectionSpec("PEP-11-01", "표준화 항목", "표준화 계획", []),
    SectionSpec("PEP-11-02", "정보화기반표준", "표준화 계획", []),
    SectionSpec("PEP-11-03", "공공기관 DB표준화 지침", "표준화 계획", []),
    SectionSpec("PEP-11-04", "전자정부 웹사이트 품질관리 지침", "표준화 계획", []),

    SectionSpec("PEP-12", "품질관리계획", None, ["품질보증계획", "품질 보증 계획"]),
    SectionSpec("PEP-13", "위험관리계획", None, []),
    SectionSpec("PEP-14", "보안대책", None, []),
    SectionSpec("PEP-15", "교육계획", None, []),
    SectionSpec("PEP-16", "발주기관 협조요청사항", None, []),
]
