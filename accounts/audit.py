# -*- coding: utf-8 -*-
"""
감사 로그 기록 헬퍼.

contracts/performance 등 다른 앱은 이 함수를 통해서만 AuditLog를 남긴다
(직접 AuditLog.objects.create를 호출하지 않음) — 기록 형식(IP 추출 방식,
organization 채우는 방식 등)을 한곳에서 통일하기 위함.
"""


def _client_ip(request):
    """프록시(X-Forwarded-For) 뒤에 있을 수 있으므로 그쪽을 먼저 본다."""
    forwarded = request.META.get("HTTP_X_FORWARDED_FOR")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR")


def log_audit(request, action, target_type, target_id=None, detail=""):
    """
    request: 로그인된 사용자의 HttpRequest (view 안에서 호출)
    action: AuditLog.ACTION_VIEW / ACTION_UPLOAD / ACTION_DELETE
    target_type: 예) 'contract_document', 'deliverable'
    target_id: 대상 레코드 PK
    """
    from .models import AuditLog

    user = request.user if request.user.is_authenticated else None
    organization = getattr(user, "organization", None) if user else None

    AuditLog.objects.create(
        user=user,
        organization=organization,
        action=action,
        target_type=target_type,
        target_id=target_id,
        detail=detail,
        ip_address=_client_ip(request),
    )
