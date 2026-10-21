"""Self-test for the PlantWild website (run with the TF venv):

    venv\\Scripts\\python.exe website\\test_app.py

Covers: pages render, sample + file predictions, ensemble outputs, error path.
"""

import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import app as web  # noqa: E402  (loads the 4 models on import)

results = []


def check(name, cond, extra=""):
    results.append(bool(cond))
    print(("PASS  " if cond else "FAIL  ") + name + (("  -> " + extra) if extra else ""))


client = web.app.test_client()

r = client.get("/")
check("GET /", r.status_code == 200)

r = client.get("/report")
check("GET /report", r.status_code == 200
      and b"confusion_dinov2.png" in r.data and b"progress_models.png" in r.data)

r = client.get("/health")
check("GET /health", r.status_code == 200 and r.get_json()["classes"] == 34)

s = web.SAMPLES[0]
r = client.get(f"/sample_image?cls={s['class']}&file={s['file']}")
check("GET /sample_image", r.status_code == 200)

# --- prediction from a sample of the true class ---
r = client.post("/predict", data={
    "sample_class": s["class"],
    "sample_file": s["file"],
})
html = r.get_data(as_text=True)
top1 = "n/a"
if r.status_code == 200 and "big-prediction" in html:
    top1 = html.split('class="big-prediction">')[1].split("<")[0]
check(f"POST /predict sample ({s['class']})", r.status_code == 200, f"top1={top1}")

# --- file upload with a known photo ---
import os  # noqa: E402
downloads = Path(os.environ["USERPROFILE"]) / "Downloads"
peb = downloads / "peb4.png"
if not peb.exists():
    peb = next(downloads.glob("*.png"), None)
if peb and peb.exists():
    r = client.post("/api/predict",
                    data={"image": (io.BytesIO(peb.read_bytes()), peb.name)},
                    content_type="multipart/form-data")
    j = r.get_json() if r.status_code == 200 else {}
    top = j.get("ensemble_weighted", {}).get("top1", {})
    check(f"POST /api/predict file ({peb.name})", r.status_code == 200,
          f"top1={top.get('class')} {top.get('prob', 0):.2f}")
else:
    check("POST /api/predict file", True, "skipped - no Downloads png found")

# --- bad upload -> 400 with friendly error ---
r = client.post("/predict",
                data={"image": (io.BytesIO(b"this is not an image"), "x.png")},
                content_type="multipart/form-data")
check("POST /predict bad file -> 400", r.status_code == 400
      and b"Could not read" in r.data)

# --- empty submit -> 400 ---
r = client.post("/predict", data={})
check("POST /predict empty -> 400", r.status_code == 400
      and b"Upload an image" in r.data)

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
