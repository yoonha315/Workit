"""
로컬 테스트용 실행 스크립트.

백엔드/프론트 연결 전, 원본 텍스트 파일과 파싱된 JSON 파일을 직접 넣어서
QA 에이전트 결과를 확인할 때 사용한다.

사용법:
    python run.py --original original.txt --parsed parsed.json --doc-type rpt

--doc-type 은 rfp / pep / rpt 중 하나.

parsed.json 형식 예시 (소제목 텍스트를 key로, 본문을 value로):
    {
        "제1절 개요": "본 사업은 ...",
        "제2절 사업의 배경 및 목적": "본 사업의 배경은 ..."
    }
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qa_agent.engine import review_section_mapping


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_report(report: dict) -> None:
    print()
    print("========== 소제목 매핑 QA 검수 ==========")
    print(f"[문서 유형]        {report['document_type']}")
    print(f"[최종 판정]        {report['review_status']}")
    print(f"[자동 진행 가능]   {'예' if report['can_auto_proceed'] else '아니오 (반려 코멘트 확인 필요)'}")
    print(f"[전체 유사도]      {report['content_similarity']}")
    print(f"[기대 소제목 수]   {report['expected_section_count']}")
    print(f"[매칭된 소제목 수] {report['matched_section_count']}")

    issues = report["issues"]
    print()
    print(f"[발견된 이슈: {len(issues)}건]")
    if not issues:
        print("- 없음")
    for issue in issues:
        code = issue.get("code") or "-"
        title = issue.get("title") or "-"
        print(f"- [{issue['issue_type']}] {code} / {title}: {issue['message']}")
        if issue.get("sample"):
            print(f"    예시: {issue['sample']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="소제목 매핑 QA 검수 에이전트 실행")
    parser.add_argument("--original", required=True, help="원본 텍스트 파일 경로 (.txt)")
    parser.add_argument("--parsed", required=True, help="파싱 결과 JSON 파일 경로 (.json)")
    parser.add_argument("--doc-type", required=True, choices=["rfp", "pep", "rpt"], help="문서 유형")
    args = parser.parse_args()

    original_text = load_text(args.original)
    parsed_sections = load_json(args.parsed)

    report = review_section_mapping(
        original_text=original_text,
        parsed_sections=parsed_sections,
        document_type=args.doc_type,
    )

    print_report(report)

    out_path = Path(args.parsed).with_suffix(".review.json")
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"[리포트 저장] {out_path}")


if __name__ == "__main__":
    main()
