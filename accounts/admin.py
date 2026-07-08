from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User, AuditLog

@admin.register(User)
class CustomUserAdmin(UserAdmin):
    fieldsets = UserAdmin.fieldsets + (
        ('추가 정보', {'fields': ('department', 'position', 'phone', 'organization')}),
    )


@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    """감사 로그는 조회만 가능 — 수정·삭제 UI를 열어두지 않는다(불변성 유지)."""
    list_display = ('created_at', 'user', 'organization', 'action', 'target_type', 'target_id', 'ip_address')
    list_filter = ('action', 'target_type', 'organization')
    search_fields = ('user__username', 'target_type', 'detail', 'ip_address')
    readonly_fields = [f.name for f in AuditLog._meta.fields]

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False
