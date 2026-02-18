import os
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors
import torch
import numpy as np
import joblib
import shutil
from io import BytesIO
from PIL import Image
from typing import List

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, APIRouter

# การเรียกใช้งาน Modules ของคุณ
# from model.DogImage import Dog, DogImage
from yolo11.yoloCrop import yoloCrop
from resnet.resnet import ResNet, ResNetBackbone, Bottleneck
from model_manager import load_model, ACTIVE_MODEL_NAME ,ACTIVE_MODEL_PATH

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pickle
import io
from sklearn.decomposition import PCA # ใช้ลดมิติเพื่อวาดรูป
from pydantic import BaseModel
import base64
from sklearn.manifold import TSNE
import seaborn as sns
from pathlib import Path

app = FastAPI()
router = APIRouter()
device = "cuda" if torch.cuda.is_available() else "cpu"
# --- Global Variables ---
# model152 = None 
model152 = ResNetBackbone(Bottleneck, [3, 8, 36, 3], embedding_size=512)
ACTIVE_MODEL_NAME = "None"
DEFAULT_MODEL_PATH = "/app/resnet/model/resne152-V01_60.pt"

# เพิ่มคำสั่ง Check ก่อนโหลดจริง (ถ้าพังจะเห็น Log ชัดเจน)
if os.path.exists(DEFAULT_MODEL_PATH):
    print(f"✅ Found model at: {DEFAULT_MODEL_PATH}")
else:
    print(f"❌ CANNOT FIND model at: {DEFAULT_MODEL_PATH}")
    # ลอง List ไฟล์รอบๆ ออกมาดูถ้าหาไม่เจอ
    print(f"Files in /app/resnet/model: {os.listdir('/app/resnet/model')}")

def load_model_engine(path: str):
    """ฟังก์ชันกลางสำหรับโหลด Weight เข้าตัวแปร Global Model"""
    global model152, ACTIVE_MODEL_NAME
    try:
        if not os.path.exists(path):
            print(f" Error: Model path {path} not found.")
            return False
        
        # โหลด state_dict
        state_dict = torch.load(path, map_location=device)
        
        # สมมติว่าคุณมีฟังก์ชันสร้างโครงสร้าง model152 ไว้แล้ว
        # model152.load_state_dict(state_dict) 
        
        # หาก num_classes ในรุ่นใหม่ไม่เท่าเดิม คุณอาจต้องโหลด meta.json มาเช็คก่อน
        # แต่เบื้องต้นถ้าโครงสร้างเหมือนเดิม ใช้บรรทัดนี้:
        model152.load_state_dict(state_dict)
        model152.to(device)
        model152.eval() # สำคัญมาก: ต้องตั้งเป็น eval mode เสมอ

        # อัปเดตชื่อรุ่นที่ใช้งานอยู่ (ดึงชื่อ Folder มาจาก Path)
        ACTIVE_MODEL_NAME = os.path.basename(os.path.dirname(path))
        print(f" Model updated to: {path}")
        return True
    except Exception as e:
        print(f"Failed to load model: {e}")
        return False
    
@router.post("/select-model")
def select_model(version: str):
    # 1. เช็คว่าถ้าส่งคำว่า "default" มา ให้ใช้ Path เริ่มต้นที่เราตั้งไว้
    if version == "default":
        model_path = DEFAULT_MODEL_PATH
    else:
        # 2. ถ้าส่งชื่อเวอร์ชันมา (เช่น V01, V02) ให้ไปหาใน checkpoints
        model_path = os.path.join(BASE_CHECKPOINT_DIR, version, "model.pth")

    # ตรวจสอบว่าไฟล์มีอยู่จริงไหม
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404, 
            detail=f"ไม่พบไฟล์โมเดล: {version} (ตรวจสอบที่: {model_path})"
        )

    # โหลดเข้าเครื่องยนต์ (Global model152)
    success = load_model_engine(model_path)
    
    if not success:
        raise HTTPException(status_code=500, detail="เกิดข้อผิดพลาดในการโหลด Weights")

    return {
        "status": "success",
        "active_model": version,
        "path": model_path
    }

@router.get("/current-model")
def current_model():
    return {
        "active_model": ACTIVE_MODEL_NAME,
        "device": str(device)
    }


@router.get("/models")
def list_models():
    models = []

    # 🔹 default model
    models.append({
        "id": "default",
        "type": "legacy",
        "path": DEFAULT_MODEL_PATH,
        "active": ACTIVE_MODEL_PATH == DEFAULT_MODEL_PATH
    })

    # 🔹 versioned models
    if os.path.exists(BASE_CHECKPOINT_DIR):
        for version_dir in sorted(os.listdir(BASE_CHECKPOINT_DIR)):
            version_path = os.path.join(BASE_CHECKPOINT_DIR, version_dir)
            model_path = os.path.join(version_path, "model.pth")
            meta_path = os.path.join(version_path, "meta.json")

            if not os.path.exists(model_path):
                continue

            info = {
                "id": version_dir,
                "type": "versioned",
                "path": model_path,
                "active": ACTIVE_MODEL_PATH == model_path
            }

            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        info["details"] = json.load(f)
                except json.JSONDecodeError:
                    info["details"] = {"error": "invalid meta.json"}

            models.append(info)

    return {"models": models}

@app.on_event("startup")
def startup_event():
    global ACTIVE_MODEL_NAME
    print(f"🚀 System starting up... Loading default model from {DEFAULT_MODEL_PATH}")
    
    # เรียกใช้ฟังก์ชัน engine ที่เราเขียนไว้เพื่อโหลด Weights เริ่มต้น
    success = load_model_engine(DEFAULT_MODEL_PATH)
    
    if success:
        # ถ้าโหลดสำเร็จ ให้ตั้งชื่อโมเดลเป็นชื่อไฟล์เริ่มต้น
        ACTIVE_MODEL_NAME = os.path.basename(DEFAULT_MODEL_PATH)
        print(f"✅ Default model loaded successfully: {ACTIVE_MODEL_NAME}")
    else:
        print(f"❌ Failed to load default model. Please check PATH1.")

## --- Dataset สำหรับโหลดรูปจาก path list ---------------------------------------------

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2(),
])
class AutoDogPipelineDatasetFromList(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

        if len(self.image_paths) == 0:
            print("Warning: No images provided to dataset")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        full_path = self.image_paths[idx]

        # --- YOLO Crop ---
        cropped_img = yoloCrop(full_path)

        if cropped_img is None:
            cropped_img = Image.open(full_path).convert("RGB")

        # --- Transform ---
        if self.transform:
            if isinstance(cropped_img, Image.Image):
                image_np = np.array(cropped_img)
            else:
                image_np = cropped_img

            augmented = self.transform(image=image_np)
            image_tensor = augmented["image"]
        else:
            image_tensor = torch.from_numpy(np.array(cropped_img)).permute(2, 0, 1).float()

        return image_tensor

def get_embedding(img_pil, model, transform, device):
    """แปลง PIL Image เป็น Embedding Vector"""
    model.eval()
    image_np = np.array(img_pil)
    augmented = transform(image=image_np)
    image_tensor = augmented['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model(image_tensor).cpu().numpy().flatten()
    return embedding

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
@app.post("/embedding-image/")
async def embedding_image(
    dog_id: int = Form(...),
    files: List[UploadFile] = File(...)
):
    try:
        results = []
        processed = 0

        for file in files:
            contents = await file.read()
            img_pil = Image.open(io.BytesIO(contents)).convert("RGB")

            # YOLO crop & Embedding logic
            cropped_img = yoloCrop(img_pil) or img_pil
            emb = get_embedding(cropped_img, model152, transform, device) # คาดว่าเป็น numpy array

            # แปลง embedding เป็น base64 string
            # ใช้ .astype(np.float32) เพื่อคุมขนาด data type ให้คงที่
            emb_bytes = emb.astype(np.float32).tobytes()
            emb_base64 = base64.b64encode(emb_bytes).decode('utf-8')

            # (Optional) อัปเดต DB ฝั่ง API ถ้าจำเป็น
            # DogImage.objects.filter(...)

            processed += 1
            results.append({
                "filename": file.filename,
                "embedding_dim": len(emb),
                "embedding_base64": emb_base64  # ส่งตัวนี้กลับไป
            })

        return {
            "dog_id": dog_id,
            "processed": processed,
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
import uuid
from fastapi import HTTPException

@app.post("/SEARCH-DOG02/")
async def search_dog02(file: UploadFile = File(...)):
    """รับรูปใหม่ แล้วค้นหา Top-3 จาก KNN ที่เซฟไว้"""

    if not os.path.exists('models/knn_latest02.joblib'):
        raise HTTPException(
            status_code=400,
            detail="KNN model not trained yet. Please run /TRAIN-KNN/ first."
        )

    # อ่านรูป
    contents = await file.read()
    img_pil = Image.open(BytesIO(contents)).convert("RGB")

    # YOLO Crop
    cropped_test_img = yoloCrop(img_pil)

    # ถ้า YOLO หาไม่เจอ → return ทันที
    if cropped_test_img is None:
        
        return {
            "status": "not_found",
            "message": "ไม่พบหมาในระบบ (YOLO ตรวจไม่พบสุนัขในภาพ)",
            "results": []
        }

    # บันทึกรูปที่ crop ได้
    save_dir = "search_history"
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"crop_{timestamp}_{uuid.uuid4().hex[:6]}.jpg"
    save_path = os.path.join(save_dir, filename)
    cropped_test_img.save(save_path)

    print(f"Saved cropped image to: {save_path}")

    # Transform → Tensor
    test_tensor = transform(image=np.array(cropped_test_img))['image']
    test_tensor = test_tensor.unsqueeze(0).to(device)

    # Extract embedding
    model152.eval()
    with torch.no_grad():
        test_embedding = model152(test_tensor).cpu().numpy()

    #  Load KNN
    knn = joblib.load("models/knn_latest02.joblib")
    filenames = joblib.load("models/labels_latest02.joblib")

    #  Search
    distances, indices = knn.kneighbors(test_embedding)
    #print("DEBUG distances:", distances)
    #print("DEBUG indices:", indices)
    
    #  Result
    unique_results = {}

    for i, idx in enumerate(indices[0]):
        dog_id = filenames[idx]
        current_distance = float(distances[0][i])
        
        # ถ้ายังไม่มี id นี้ใน dict หรือเจอตัวที่ระยะทาง (distance) น้อยกว่าเดิม
        if dog_id not in unique_results or current_distance < unique_results[dog_id]:
            unique_results[dog_id] = current_distance

    # นำข้อมูลจาก dict มาแปลงเป็น list ของ objects
    sorted_results = [
        {"dog_id": dog_id, "distance": dist} 
        for dog_id, dist in unique_results.items()
    ]

    # เรียงลำดับตาม distance จากน้อยไปมาก และเอาแค่ 5 ตัวแรก
    sorted_results = sorted(sorted_results, key=lambda x: x["distance"])[:5]

    # เพิ่มลำดับ rank หลังจากกรองและเรียงเสร็จแล้ว
    final_results = []
    for i, res in enumerate(sorted_results):
        final_results.append({
            "rank": i + 1,
            "dog_id": res["dog_id"],
            "distance": res["distance"]
        })

    return {"results": final_results}

import asyncio
import json
import threading
import os
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import LabelEncoder
from middleware.auth import verify_token
from resnet.train import FaceModelTrainer
from resnet.DataLoader import get_dataloaders

# --- 1. เพิ่ม CORS Middleware (แก้ปัญหาเชื่อมต่อไม่ได้) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Queue สำหรับเก็บ SSE Clients
train_status_queues = set()

async def broadcast_status(data):
    """ส่งข้อมูลเข้า Queue ของ SSE ทุกอัน"""
    if train_status_queues:
        if isinstance(data, str):
            data = {"status": data}
        formatted_msg = f"data: {json.dumps(data)}\n\n"
        await asyncio.gather(*[q.put(formatted_msg) for q in train_status_queues])

# ---   ฟังก์ชันเทรนที่ทำงานใน Thread แยก ---

def get_next_version(base_dir: str):
    os.makedirs(base_dir, exist_ok=True)

    versions = [
        d for d in os.listdir(base_dir)
        if d.startswith("v") and d[1:].isdigit()
    ]

    if not versions:
        return "v001"

    latest = max(int(v[1:]) for v in versions)
    return f"v{latest + 1:03d}"
# สร้างโมเดล ResNet152 สำหรับการฝึกสอน
train_model = ResNetBackbone(Bottleneck, [3, 8, 36, 3],embedding_size=512).to(device)

class TrainingItem(BaseModel):
    image_path: str
    label: int

class RetrainRequest(BaseModel):
    data: List[TrainingItem]
    model_type: str
def background_train_task(loop, training_data: List[TrainingItem]):
    
    def sse_callback(progress_data):
        # ส่งข้อมูลกลับไปหา Event Loop ของ FastAPI
        asyncio.run_coroutine_threadsafe(broadcast_status(progress_data), loop)

    try:
        if not training_data:
            sse_callback("⚠️ ไม่พบข้อมูลรูปภาพที่ส่งมาจาก Django")
            return

        sse_callback(f"📦 ได้รับข้อมูล {len(training_data)} รายการ เตรียมการ Crop...")

        cropped_img_list = []
        labels = [] 
        
        # 1. ประมวลผลรูปภาพจาก Payload ที่ได้รับ
        for item in training_data:
            full_path = item.image_path # ใช้ path ที่ส่งมาจาก Django
            
            if not os.path.exists(full_path):
                print(f"File not found: {full_path}")
                continue

            # ประมวลผลรูปภาพ (YOLO Crop)
            # หมายเหตุ: result ควรเป็น Image object หรือ Path ที่ crop แล้วตามที่ get_dataloaders ต้องการ
            result = yoloCrop(full_path) 
            
            if result is None:
                continue

            cropped_img_list.append(result)
            labels.append(item.label)

        if not cropped_img_list:
            sse_callback("❌ ไม่มีรูปภาพที่ใช้งานได้หลังการ Crop (YOLO หาใบหน้าไม่เจอ)")
            return

        # 2. Label Encoding
        sse_callback("กำลังจัดเตรียม Labels...")
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels) 
        actual_num_classes = len(le.classes_) 

        # 3. สร้าง DataLoader (ส่งรูปที่ Crop แล้วเข้าไป)
        train_loader, _ = get_dataloaders(
            train_path=cropped_img_list,
            image_ids=encoded_labels, 
            batch_size=min(32, len(cropped_img_list)) # ปรับ batch_size ไม่ให้เกินจำนวนรูป
        )

        # 4. ตั้งค่า Path และ Version
        BASE_CHECKPOINT_DIR = "checkpoints/resnet152"
        version = get_next_version(BASE_CHECKPOINT_DIR)
        save_path = os.path.join(BASE_CHECKPOINT_DIR, version)
        os.makedirs(save_path, exist_ok=True)

        # 5. เริ่มการฝึกสอน
        sse_callback(f"🚀 เริ่มเทรน Model {version} ({actual_num_classes} คลาส)...")
        
        # สมมติว่า train_model ถูกนิยามไว้ข้างนอก และเราต้องการปรับ output layer ตามจำนวน class จริง
        # model = build_model(num_classes=actual_num_classes) 
        
        trainer = FaceModelTrainer(
            model=train_model, 
            train_loader=train_loader,
            device=device,
            num_classes=actual_num_classes,
            embedding_size=512
        )
        
        trainer.train(epochs=3, save_path=save_path, progress_callback=sse_callback)

        # 6. บันทึก Model และ Metadata
        torch.save(train_model.state_dict(), os.path.join(save_path, "model.pth"))

        meta = {
            "model": "resnet152",
            "version": version,
            "trained_at": datetime.now().isoformat(),
            "num_classes": int(actual_num_classes),
            "classes": [int(c) for c in le.classes_], 
            "num_images": len(cropped_img_list),
            "epochs": 3,
            "device": str(device)
        }

        with open(os.path.join(save_path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        sse_callback(f"✅ การฝึกสอน {version} เสร็จสมบูรณ์!")

    except Exception as e:
        sse_callback(f"❌ Error: {str(e)}")
        print(f"Error during training: {e}")

# --- 3. API Endpoints ---

@app.get("/train-progress")
async def stream_training_progress():
    """Endpoint สำหรับเชื่อมต่อ Log (SSE)"""
    queue = asyncio.Queue()
    train_status_queues.add(queue)
    
    async def event_generator():
        try:
            yield f"data: {json.dumps({'status': '📡 เชื่อมต่อกับ Server สำเร็จ'})}\n\n"
            while True:
                data = await queue.get()
                yield data
        except asyncio.CancelledError:
            train_status_queues.remove(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
from pydantic import BaseModel
from typing import List


@app.post("/retrain-model-face")
async def retrain(request: RetrainRequest, payload=Depends(verify_token)):
    """API สำหรับเริ่มเทรน โดยรับข้อมูล List ของรูปภาพและ Label"""
    
    loop = asyncio.get_running_loop()
    
    # ส่ง request.data (List ของ TrainingItem) เข้าไปใน background task
    thread = threading.Thread(
        target=background_train_task, 
        args=(loop, request.data)
    )
    thread.start()
    
    return {
        "message": "Training started in background",
        "received_items": len(request.data)
    }

## -----------------------test model knn classification ---------------------------


class EmbeddingItem(BaseModel):
    dog_id: int
    embedding_b64: str

class TrainRequest(BaseModel):
    data: List[EmbeddingItem]

@app.post("/tiger_knnTrain/")
async def train_knn02(request: TrainRequest):
    X = []
    y = []

    # 2. แกะข้อมูลจาก Request Body
    for item in request.data:
        try:
            # แปลง Base64 กลับเป็น bytes -> แล้วแปลงเป็น numpy array
            binary_data = base64.b64decode(item.embedding_b64)
            emb = np.frombuffer(binary_data, dtype=np.float32)

            # Sanity check
            if emb.ndim != 1 or emb.shape[0] == 0:
                continue

            X.append(emb)
            y.append(item.dog_id)
        except Exception as e:
            print(f"Error processing dog_id {item.dog_id}: {e}")
            continue

    if len(X) == 0:
        raise HTTPException(status_code=400, detail="No valid embeddings provided")

    # 3. เตรียมข้อมูลและ Train
    X_array = np.vstack(X)

    # ใช้ Cosine Metric ตามเดิม
    knn = NearestNeighbors(
        n_neighbors=len(X_array),
        # n_neighbors=min(3, len(X_array)),
        metric="cosine"
    )
    knn.fit(X_array)

    # 4. บันทึก Model
    os.makedirs("models", exist_ok=True)
    joblib.dump(knn, "models/knn_latest02.joblib")
    joblib.dump(y, "models/labels_latest02.joblib")

    return {
        "status": "success",
        "total_embeddings_trained": len(X_array)
    }

def create_plot_64(X_transformed, y, title, xlabel, ylabel):
    # ใช้หมวด Agg จะช่วยลดปัญหาเรื่อง Thread ของ Tkinter
    plt.figure(figsize=(10, 7))
    try:
        scatter = plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y, cmap='viridis', edgecolors='k', alpha=0.7)
        plt.colorbar(scatter, label='Dog ID')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        return img_b64
    finally:
        # สำคัญมาก: ต้องปิด plot ทุกครั้งเพื่อคืน Memory และเลี่ยงปัญหา Thread
        plt.clf()
        plt.close('all')

@app.post("/test-knn/")
async def test_knn(request: TrainRequest):
    try:
        embeddings = []
        labels = []

        #  Decode ข้อมูล
        for item in request.data:
            binary_data = base64.b64decode(item.embedding_b64)
            vector = np.frombuffer(binary_data, dtype=np.float32) 
            embeddings.append(vector)
            labels.append(item.dog_id)

        X = np.array(embeddings)
        y = np.array(labels)

        if len(X) < 2:
            raise ValueError("Data points must be at least 2 for visualization.")

        # t-SNE เหมาะกับการดูการกระจายกลุ่ม (Cluster) ของข้อมูลมิติสูง
        perplexity = min(20, len(X) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_tsne = tsne.fit_transform(X)
        tsne_image = create_plot_64(X_tsne, y, "t-SNE Visualization of Dog Embeddings", "t-SNE dimension 1", "t-SNE dimension 2")

        # ---  KNN Confusion Matrix ---
        knn = KNeighborsClassifier(n_neighbors=2)
        knn.fit(X, y)
        y_pred = knn.predict(X)

        #  คำนวณค่า Accuracy
        acc = accuracy_score(y, y_pred)

        #  สร้าง Confusion Matrix (ส่วนเดิมของคุณ)
        cm = confusion_matrix(y, y_pred)
        unique_labels = np.unique(y)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                     xticklabels=unique_labels, 
                    yticklabels=unique_labels)
        plt.title(f"KNN Confusion Matrix (Accuracy: {acc:.2f})") # แสดง Accuracy ใน Title ด้วยก็ได้ครับ
        plt.ylabel('Actual Dog ID')
        plt.xlabel('Predicted Dog ID')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        knn_matrix_image = base64.b64encode(buf.read()).decode('utf-8')
        plt.clf()
        plt.close('all')

        return {
            "status": "success",
            "accuracy": float(acc), # ส่งค่า accuracy กลับไป (แปลงเป็น float ปกติ)
            "tsne_plot": tsne_image,
            "knn_matrix": knn_matrix_image,
            "model_name": ACTIVE_MODEL_NAME
        }

    except Exception as e:
        plt.close('all')
        raise HTTPException(status_code=500, detail=str(e))
    
app.include_router(router)
