from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# ── xView vector stores ───────────────────────────────────────────────────────
XVIEW_DINO_INDEX_PATHS = {
    128: BASE_DIR / "database" / "VectorStores" / "dino" / "dino_128.index",
    256: BASE_DIR / "database" / "VectorStores" / "dino" / "dino_256.index",
    512: BASE_DIR / "database" / "VectorStores" / "dino" / "dino_512.index",
}
XVIEW_DINO_METADATA_DB = BASE_DIR / "database" / "VectorStores" / "dino" / "metadata.db"

XVIEW_CLIP_INDEX_PATH  = BASE_DIR / "database" / "VectorStores" / "clip" / "clip_256.index"
XVIEW_CLIP_METADATA_DB = BASE_DIR / "database" / "VectorStores" / "clip" / "clip_metadata.db"

# ── DIOR vector stores (paths — populate once the index files are ready) ──────
DIOR_DINO_INDEX_PATHS = {
    128: BASE_DIR / "database" / "VectorStores" / "dior_dino" / "dino_128.index",
    256: BASE_DIR / "database" / "VectorStores" / "dior_dino" / "dino_256.index",
    512: BASE_DIR / "database" / "VectorStores" / "dior_dino" / "dino_512.index",
}
DIOR_DINO_METADATA_DB  = BASE_DIR / "database" / "VectorStores" / "dior_dino" / "metadata.db"

DIOR_CLIP_INDEX_PATH   = BASE_DIR / "database" / "VectorStores" / "dior_clip" / "clip_256.index"
DIOR_CLIP_METADATA_DB  = BASE_DIR / "database" / "VectorStores" / "dior_clip" / "clip_metadata.db"

# Backwards-compat aliases (used by live_search / annotate routers)
DINO_INDEX_PATHS = XVIEW_DINO_INDEX_PATHS
DINO_METADATA_DB = XVIEW_DINO_METADATA_DB
CLIP_INDEX_PATH  = XVIEW_CLIP_INDEX_PATH
CLIP_METADATA_DB = XVIEW_CLIP_METADATA_DB

# Source satellite image directories (for chip extraction)
XVIEW_IMAGES_DIR = BASE_DIR / "database" / "SatelliteImages" / "xView" / "train_images" / "train_images"
DIOR_IMAGES_DIRS = [
    BASE_DIR / "database" / "SatelliteImages" / "DIOR" / "train" / "images",
    BASE_DIR / "database" / "SatelliteImages" / "DIOR" / "test" / "images",
]

# Temp output directory for YOLO datasets
TEMP_DATASET_DIR = BASE_DIR / "backend" / "tmp"
TEMP_DATASET_DIR.mkdir(parents=True, exist_ok=True)
