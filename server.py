import os
import shutil
import subprocess
from datetime import datetime
import numpy as np
from PIL import Image
from osgeo import gdal

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

CODE_ROOT = r"C:\Users\SamMorgan\Downloads\Coseismic-landslide-detection-main\Coseismic-landslide-detection-main\landsldie_detection_code\code"

INPUT_TIF_PATH = os.path.join(CODE_ROOT, "data", "test", "0", "test.tif")
PATCH_DIR = os.path.join(CODE_ROOT, "data", "test", "1")
PREDICT_DIR = os.path.join(CODE_ROOT, "data", "test", "3")
PREVIEW_DIR = os.path.join(CODE_ROOT, "data", "test", "4")
PREVIEW_PNG_PATH = os.path.join(PREVIEW_DIR, "mosaic_preview.png")
ORIGINAL_PNG_PATH = os.path.join(PREVIEW_DIR, "original_preview.png")

REPORT_DIR = os.path.join(CODE_ROOT, "Unet_Resnet", "report")

# 主输出目录
OUTPUTS_DIR = os.path.join(CODE_ROOT, "web_outputs")
# 归档目录：唯一的永久保存位置
ARCHIVE_ROOT = os.path.join(OUTPUTS_DIR, "archive")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(ARCHIVE_ROOT, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 【关键修改1】同时挂载 archive 目录为静态文件，让前端可以直接访问
app.mount("/outputs/archive", StaticFiles(directory=ARCHIVE_ROOT), name="archive")
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")


def run_script(script_abs_path: str, cwd: str):
    print(f"[INFO] Running script: {script_abs_path}")
    print(f"[INFO] Working directory: {cwd}")

    proc = subprocess.run(
        ["python", script_abs_path],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='ignore'
    )

    if proc.stdout:
        print(f"[STDOUT]\n{proc.stdout}")
    if proc.stderr:
        print(f"[STDERR]\n{proc.stderr}")

    if proc.returncode != 0:
        error_msg = (
            f"Script failed: {script_abs_path}\n"
            f"Return code: {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )
        print(f"[ERROR] {error_msg}")
        raise RuntimeError(error_msg)

    print(f"[SUCCESS] Script completed: {script_abs_path}")


def convert_tif_to_png(tif_path, png_path):
    """Convert GeoTIFF to PNG preview (with downsampling)"""

    def _normalize_band(band):
        band_min = band.min()
        band_max = band.max()
        if band_max == band_min:
            return np.zeros_like(band, dtype=np.uint8)
        normalized = (band - band_min) / (band_max - band_min) * 255
        return normalized.astype(np.uint8)

    try:
        print(f"\n[INFO] ========== Converting TIF to PNG ==========")
        print(f"[INFO] Input TIF: {tif_path}")

        ds = gdal.Open(tif_path)
        if ds is None:
            print(f"[ERROR] Cannot open TIFF file: {tif_path}")
            return False

        band_count = ds.RasterCount
        width = ds.RasterXSize
        height = ds.RasterYSize
        print(f"[INFO] Original size: {width}x{height}, Bands: {band_count}")

        max_size = 2048
        scale_factor = 1.0
        if width > max_size or height > max_size:
            scale_factor = max_size / max(width, height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            print(f"[INFO] Image too large, scaling to {new_width}x{new_height}")
        else:
            new_width, new_height = width, height

        if band_count >= 3:
            b1 = ds.GetRasterBand(1).ReadAsArray()
            b2 = ds.GetRasterBand(2).ReadAsArray()
            b3 = ds.GetRasterBand(3).ReadAsArray()

            if b1.dtype == np.float32 or b1.dtype == np.float64 or b1.max() > 255:
                b1 = _normalize_band(b1)
                b2 = _normalize_band(b2)
                b3 = _normalize_band(b3)
            else:
                b1 = np.clip(b1, 0, 255).astype(np.uint8)
                b2 = np.clip(b2, 0, 255).astype(np.uint8)
                b3 = np.clip(b3, 0, 255).astype(np.uint8)

            if scale_factor < 1.0:
                img_b1 = Image.fromarray(b1).resize((new_width, new_height), Image.LANCZOS)
                img_b2 = Image.fromarray(b2).resize((new_width, new_height), Image.LANCZOS)
                img_b3 = Image.fromarray(b3).resize((new_width, new_height), Image.LANCZOS)
                b1, b2, b3 = np.array(img_b1), np.array(img_b2), np.array(img_b3)

            rgb = np.dstack((b1, b2, b3))
        elif band_count == 1:
            band = ds.GetRasterBand(1).ReadAsArray()
            if band.dtype == np.float32 or band.dtype == np.float64 or band.max() > 255:
                band = _normalize_band(band)
            else:
                band = np.clip(band, 0, 255).astype(np.uint8)

            if scale_factor < 1.0:
                img_band = Image.fromarray(band).resize((new_width, new_height), Image.LANCZOS)
                band = np.array(img_band)

            rgb = np.stack([band, band, band], axis=-1)
        else:
            print(f"[ERROR] Unsupported band count: {band_count}")
            ds = None
            return False

        img = Image.fromarray(rgb)
        img.save(png_path, quality=95)
        print(f"[SUCCESS] PNG saved to: {png_path}")
        ds = None
        return True
    except Exception as e:
        print(f"[ERROR] Failed to convert TIFF to PNG: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def find_latest_docx():
    if not os.path.exists(REPORT_DIR):
        return None
    candidates = []
    for name in os.listdir(REPORT_DIR):
        if name.startswith("landslide_report_") and name.endswith(".docx"):
            full = os.path.join(REPORT_DIR, name)
            candidates.append((os.path.getmtime(full), full))
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1]


def cleanup_middle():
    """只清理中间处理数据，不触碰归档文件夹"""
    for d in [PATCH_DIR, PREDICT_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)

    # 清理临时预览文件夹
    if os.path.exists(PREVIEW_DIR):
        try:
            os.remove(PREVIEW_PNG_PATH)
        except FileNotFoundError:
            pass
        try:
            os.remove(ORIGINAL_PNG_PATH)
        except FileNotFoundError:
            pass
    else:
        os.makedirs(PREVIEW_DIR, exist_ok=True)


@app.post("/api/segment")
async def segment(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".tif", ".tiff"]:
        raise HTTPException(status_code=400, detail="Only .tif/.tiff supported")

    # 1. 生成唯一的时间戳 ID
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n[INFO] ========== Starting new run: {run_timestamp} ==========")

    # 2. 创建本次运行的归档文件夹（唯一保存位置）
    run_archive_dir = os.path.join(ARCHIVE_ROOT, run_timestamp)
    os.makedirs(run_archive_dir, exist_ok=True)
    print(f"[INFO] Archive directory: {run_archive_dir}")

    cleanup_middle()

    # 3. 保存上传的原始文件
    os.makedirs(os.path.dirname(INPUT_TIF_PATH), exist_ok=True)
    with open(INPUT_TIF_PATH, "wb") as f:
        f.write(await file.read())

    # 归档原始 TIF
    archived_original_tif = os.path.join(run_archive_dir, f"original_{file.filename}")
    shutil.copyfile(INPUT_TIF_PATH, archived_original_tif)
    print(f"[SUCCESS] Original TIF archived to: {archived_original_tif}")

    try:
        print("\n" + "=" * 60)
        print("Step 1/3: Cropping image...")
        print("=" * 60)
        run_script(os.path.join(CODE_ROOT, "code_0", "gdal_crop.py"), cwd=os.path.join(CODE_ROOT, "code_0"))

        print("\n" + "=" * 60)
        print("Step 2/3: Running segmentation...")
        print("=" * 60)
        run_script(os.path.join(CODE_ROOT, "Unet_Resnet", "predict_gdal.py"),
                   cwd=os.path.join(CODE_ROOT, "Unet_Resnet"))

        print("\n" + "=" * 60)
        print("Step 3/3: Mosaicking results...")
        print("=" * 60)
        run_script(os.path.join(CODE_ROOT, "code_0", "gdal_combine.py"), cwd=os.path.join(CODE_ROOT, "code_0"))

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    if not os.path.exists(PREVIEW_PNG_PATH):
        raise HTTPException(status_code=500, detail="Result image not generated")

    # 【关键修改2】只保存到 archive，不再复制到 web_outputs 根目录
    # 4. 归档分割结果
    archived_mosaic = os.path.join(run_archive_dir, "segmentation_result.png")
    shutil.copyfile(PREVIEW_PNG_PATH, archived_mosaic)
    print(f"[SUCCESS] Segmentation result archived to: {archived_mosaic}")

    # 5. 生成并归档原始影像预览
    archived_original_png = os.path.join(run_archive_dir, "original_preview.png")
    original_url = None

    if os.path.exists(INPUT_TIF_PATH):
        if convert_tif_to_png(INPUT_TIF_PATH, ORIGINAL_PNG_PATH):
            shutil.copyfile(ORIGINAL_PNG_PATH, archived_original_png)
            print(f"[SUCCESS] Original preview archived to: {archived_original_png}")
            # 【关键修改3】直接返回 archive 中的图片 URL
            original_url = f"/outputs/archive/{run_timestamp}/original_preview.png"

    print(f"\n[SUCCESS] ========== Run {run_timestamp} completed ==========")

    return {
        "previewUrl": f"/outputs/archive/{run_timestamp}/segmentation_result.png",
        "originalUrl": original_url,
        "archiveId": run_timestamp
    }


@app.post("/api/report")
async def report():
    try:
        run_script(os.path.join(CODE_ROOT, "Unet_Resnet", "LLM.py"), cwd=os.path.join(CODE_ROOT, "Unet_Resnet"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latest = find_latest_docx()
    if not latest:
        raise HTTPException(status_code=500, detail="No report docx found")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_docx = os.path.join(OUTPUTS_DIR, f"landslide_report_latest_{ts}.docx")
    shutil.copyfile(latest, out_docx)

    return FileResponse(
        out_docx,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=os.path.basename(out_docx),
    )