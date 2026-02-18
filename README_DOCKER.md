# FindMyDog Docker Implementation Guide

## 1. Prerequisites
- Docker & Docker Compose installed

## 2. Setup & Run

1. **Create .env file:**
   ```bash
   cp .env.example .env
   ```

2. **Build and Run:**
   ```bash
   docker-compose up --build
   ```

   - This will build the images for Django (`web`) and FastAPI (`ai`).
   - It will start PostgreSQL (`db`) and Redis (`redis`).
   - It automatically runs `python manage.py migrate` and `python manage.py collectstatic` for Django.

3. **Access the services:**
   - **Django Web:** [http://localhost:8000](http://localhost:8000)
   - **FastAPI AI:** [http://localhost:8080](http://localhost:8080) (Docs: [http://localhost:8080/docs](http://localhost:8080/docs))

## 3. Networking Logic & Integration

### How Django talks to FastAPI?
In the Docker network `findmydog_network`:
- The FastAPI service is named `ai`.
- Django can reach it via `http://ai:8000`.

**Example Usage in Django views:**
```python
import requests

response = requests.post("http://ai:8000/process-image/", files=...)
```

### Shared Volumes (Media)
Both services share the `./media` directory on the host, mapped to `/app/media` in the containers.
- When Django saves an uploaded image to `/app/media/dogs/dog1.jpg`, it appears instantly in the FastAPI container at the same path.
- This allows FastAPI to process images directly from the filesystem without needing to transfer binary data over HTTP (though HTTP is still used to trigger the task).
