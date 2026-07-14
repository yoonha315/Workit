import os
from celery import Celery
from celery.schedules import crontab

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

app = Celery('workit')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()

# Redis 3.x 호환
app.conf.broker_transport_options = {
    'visibility_timeout': 3600,
    'socket_connect_timeout': 10,
}

# celerybeat-schedule.* 파일이 프로젝트 루트에 그대로 쌓이지 않도록 전용 폴더에 저장한다.
_VAR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'var')
os.makedirs(_VAR_DIR, exist_ok=True)
app.conf.beat_schedule_filename = os.path.join(_VAR_DIR, 'celerybeat-schedule')

app.conf.beat_schedule = {
    'check-deliverable-deadlines': {
        'task': 'performance.tasks.check_deadlines',
        'schedule': crontab(hour=10, minute=0),  # 매일 오전 10시
    },
}