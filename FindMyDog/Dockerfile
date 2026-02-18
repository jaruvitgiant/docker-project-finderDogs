# ใช้ Python 3.12 ตามที่แนะนำไปก่อนหน้านี้เพื่อให้รองรับ django-tailwind
FROM python:3.12-slim

WORKDIR /app

# # 1. Install System Dependencies (ครบถ้วนสำหรับ Django และ PostgreSQL)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     netcat-openbsd \
#     postgresql-client \
#     libpq-dev \
#     libjpeg-dev \
#     zlib1g \
#     gcc \
#     python3-dev \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    netcat-openbsd \
    postgresql-client \
    libpq-dev \
    libjpeg-dev \
    zlib1g \
    gcc \
    python3-dev \
    build-essential \
    cron \
    && rm -rf /var/lib/apt/lists/*

# 2. จัดการเรื่อง User
RUN useradd -m -u 1000 django && \
    mkdir -p /app/staticfiles /app/media && \
    chown -R django:django /app

# 3. ติดตั้ง Python Dependencies (ติดตั้งตรงๆ ใน stage นี้เลย)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Copy Project Files
COPY --chown=django:django . .

# ตรวจสอบว่ามีไฟล์ entrypoint.sh อยู่จริง
RUN chmod +x /app/entrypoint.sh

# 5. Environment Settings
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER django

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
# ปรับชื่อโปรเจกต์ (findmydog) ให้ตรงกับชื่อโฟลเดอร์ที่มี wsgi.py ของคุณ
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "3", "findmydog.wsgi:application"]