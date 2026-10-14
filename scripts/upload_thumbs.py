"""Upload thumbs.zip (from the Kaggle notebook) to a Supabase Storage bucket, one file per chip.

Usage:  python scripts/upload_thumbs.py /path/to/thumbs.zip   (or a folder of .webp files)
Reads SUPABASE_URL, SUPABASE_SERVICE_KEY, STORAGE_BUCKET from backend/.env (or the environment).
Safe to re-run: existing files are overwritten (x-upsert).
"""
import os, sys, zipfile, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / "backend" / ".env")
except ImportError:
    pass

URL = os.environ["SUPABASE_URL"].rstrip("/")
KEY = os.environ["SUPABASE_SERVICE_KEY"]
BUCKET = os.getenv("STORAGE_BUCKET", "thumbs")
zpath = sys.argv[1]

src = Path(zpath)
if src.is_dir():
    names = sorted(str(p) for p in src.glob("*.webp"))
    read = lambda n: Path(n).read_bytes()
else:
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.endswith(".webp")]
    read = lambda n: zipfile.ZipFile(zpath).read(n)   # own handle per call: thread-safe
print(f"{len(names)} files -> {URL}/storage/v1/object/public/{BUCKET}/")

def upload(name):
    data = read(name)
    req = urllib.request.Request(
        f"{URL}/storage/v1/object/{BUCKET}/{Path(name).name}", data=data, method="POST",
        headers={"Authorization": f"Bearer {KEY}", "apikey": KEY,
                 "Content-Type": "image/webp", "x-upsert": "true"})
    for attempt in range(3):
        try:
            urllib.request.urlopen(req, timeout=30).read(); return None
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 403, 404): return f"{name}: HTTP {e.code} {e.read()[:120]!r}"
        except Exception as e:
            err = str(e)
    return f"{name}: failed"

fails, done = [], 0
with ThreadPoolExecutor(16) as ex:
    for r in ex.map(upload, names):
        done += 1
        if r: fails.append(r)
        if done % 2000 == 0: print(f"{done}/{len(names)} (failures so far: {len(fails)})", flush=True)
print(f"done. uploaded {len(names) - len(fails)} / {len(names)}")
for f in fails[:5]: print("FAIL", f)
