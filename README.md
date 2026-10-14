# Satellite Visual Search

Search satellite imagery by example. Upload an image chip, and the system finds visually
similar chips across the xView dataset, plotted on a 3D globe, with YOLO dataset export and
VLM-assisted analysis.

```
frontend/   React + Vite + Tailwind/shadcn UI (globe, panels)
backend/    FastAPI + DINOv2 embeddings; search runs on Postgres + pgvector (Supabase)
notebooks/  build_demo_dataset.ipynb — builds the 50k-chip demo dataset on Kaggle
scripts/    upload_thumbs.py — uploads chip thumbnails to Supabase Storage
```

## How it works
1. `notebooks/build_demo_dataset.ipynb` cuts 128px chips from xView, embeds them with
   `facebook/dinov2-small` (384-d, CLS token), and loads them into a Supabase `chips` table
   (`vector(384)` + HNSW index) plus thumbnails into a public Storage bucket.
2. The backend embeds the uploaded query chip and runs `ORDER BY embedding <=> query` in SQL.
3. Without `DATABASE_URL`, the backend falls back to the local FAISS indexes
   (`database/VectorStores/…`), which are not part of this repo.

## Run locally
**Backend**
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in DATABASE_URL, SUPABASE_URL, etc.
uvicorn main:app --reload     # http://localhost:8000  (docs at /docs)
```
**Frontend**
```bash
cd frontend
npm install
cp .env.example .env.local    # optional; VITE_API_BASE defaults to http://localhost:8000
npm run dev                   # http://localhost:5173
```

## Configuration (backend/.env)
| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Supabase Postgres (session pooler). Enables cloud search. |
| `SUPABASE_URL`, `STORAGE_BUCKET` | Where result thumbnails are served from |
| `DEMO_CHIP_SIZE` | Chip size in the demo dataset (128) |
| `ALLOWED_ORIGINS` | CORS origins, comma-separated |
| `MAPTILER_KEY` | Live satellite map tiles |
| `OPENROUTER_API_KEY` | VLM chat / reports |
| `DEVICE` | Force `cpu` or `cuda` (auto-detected by default) |

## Cloud demo limits
Only 128px xView chips are loaded (50k). Text (CLIP) search and the DIOR dataset are
unavailable in this mode.

## Deploy
Backend: `backend/Dockerfile` (CPU torch). Frontend: build with `npm run build` in `frontend/`
and set `VITE_API_BASE` to the deployed API URL.
