# ====================================
# FastAPI AI Service (Robust Version)
# ====================================
FROM python:3.10-slim

WORKDIR /app

# 1. Install System Dependencies 
RUN apt-get update && apt-get install -y --no-install-recommends \
    netcat-openbsd \
    curl \
    gcc \
    python3-dev \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    # เพิ่ม build-essential เผื่อบางแพ็กเกจต้อง compile
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. จัดการเรื่อง User และ Permissions
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/media && \
    chown -R appuser:appuser /app

# 3. ติดตั้ง Python Dependencies
COPY requirements.txt .

# --- จุดสำคัญ: ปรับปรุงการติดตั้งเพื่อกันพัง ---
# - เพิ่ม --default-timeout เพราะ Torch ไฟล์ใหญ่มาก (เน็ตอาจตัด)
# - แยกการลง numpy ออกมาเพื่อคุมเวอร์ชันไม่ให้ตีกับ Ultralytics
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir opencv-python-headless && \
    pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# 4. ตั้งค่า Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:${PATH}"

# 5. Copy Application
COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

# เพิ่ม start-period เป็น 120s เพราะ AI โหลด Model นานมาก เดี๋ยวโดน Docker สั่งปิดก่อน
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ตรวจสอบชื่อไฟล์รันให้ชัวร์ (runserver:app หรือ main:app)
CMD ["uvicorn", "app_ubuntu:app", "--host", "0.0.0.0", "--port", "8000"]