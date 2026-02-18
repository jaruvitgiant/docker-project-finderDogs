import requests
import logging
import base64

from django.conf import settings
from django.http import JsonResponse
from .models import DogImage
import os
import logging

# apiurl=http://ai:8000/
# apiurl = os.getenv('AI_SERVICE_URL', 'http://localhost:8080')
apiurl = os.getenv("AI_SERVICE_URL001", "http://ai:8000")
# url = f"{api}/tiger_knnTrain/"

# ตรวจสอบว่า apiurl ไม่เป็น None ก่อนจะใช้ endswith
if apiurl and not apiurl.endswith('/'):
    apiurl += '/'
TRAIN_ENDPOINT = "/tiger_knnTrain/"

# เพิ่มบรรทัดนี้เพื่อตั้งค่า logger
logger = logging.getLogger(__name__)

def trainKNN():
    """
    เรียก FastAPI เพื่อ train KNN จาก embeddings ใน DogImage
    """

    # 1. ดึงรูปที่มี embedding
    images = DogImage.objects.exclude(embedding_binary__isnull=True)

    train_data = []

    for img in images:
        embedding_b64 = base64.b64encode(
            img.embedding_binary
        ).decode("utf-8")

        train_data.append({
            "dog_id": img.dog_id,
            "embedding_b64": embedding_b64
        })

    if not train_data:
        return JsonResponse(
            {"status": "error", "message": "ไม่มี embedding ในฐานข้อมูล"},
            status=400
        )

    url = apiurl + TRAIN_ENDPOINT

    try:
        response = requests.post(
            url,
            json={"data": train_data},
            timeout=300
        )

        if response.status_code != 200:
            logger.error(f"Train KNN failed: {response.status_code} {response.text}")
            return JsonResponse(
                {
                    "status": "error",
                    "code": response.status_code,
                    "detail": response.text
                },
                status=response.status_code
            )

        logger.info("Train KNN success")
        return JsonResponse(response.json(), status=200)

    except requests.exceptions.RequestException as e:
        logger.exception("Cannot connect to FastAPI")
        return JsonResponse(
            {
                "status": "error",
                "detail": str(e)
            },
            status=500
        )
