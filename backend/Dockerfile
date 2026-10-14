FROM python:3.12-slim
WORKDIR /app/backend
# CPU-only torch keeps the image small (no GPU needed for single-image queries)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Pre-download DINOv2 so the first request isn't slow
RUN python -c "from transformers import AutoModel, AutoImageProcessor as P; P.from_pretrained('facebook/dinov2-small'); AutoModel.from_pretrained('facebook/dinov2-small')"
ENV PORT=8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
