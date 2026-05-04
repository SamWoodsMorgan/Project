import os
import sys
import glob
import numpy as np
from osgeo import gdal
from math import ceil
from tqdm import tqdm
from PIL import Image

# 设置标准输出编码为 UTF-8
if sys.platform == 'win32':
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

gdal.UseExceptions()
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"


def GetExtent(infile):
    ds = gdal.Open(infile)
    geotrans = ds.GetGeoTransform()
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    min_x = geotrans[0]
    max_y = geotrans[3]
    max_x = geotrans[0] + xsize * geotrans[1]
    min_y = geotrans[3] + ysize * geotrans[5]
    ds = None
    return min_x, max_y, max_x, min_y


def RasterMosaic(file_list, outpath):
    if not file_list:
        print("[ERROR] No tif files found!")
        return

    min_x, max_y, max_x, min_y = GetExtent(file_list[0])
    for infile in file_list[1:]:
        mx, my, xx, yy = GetExtent(infile)
        min_x = min(min_x, mx)
        min_y = min(min_y, yy)
        max_x = max(max_x, xx)
        max_y = max(max_y, my)

    ref_ds = gdal.Open(file_list[0])
    geotrans = ref_ds.GetGeoTransform()
    pixel_width = geotrans[1]
    pixel_height = geotrans[5]
    proj = ref_ds.GetProjection()
    band_count = ref_ds.RasterCount
    data_type = ref_ds.GetRasterBand(1).DataType

    cols = ceil((max_x - min_x) / abs(pixel_width))
    rows = ceil((max_y - min_y) / abs(pixel_height))

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        outpath, cols, rows, band_count, data_type,
        options=["TILED=YES", "COMPRESS=PACKBITS"]
    )
    out_ds.SetProjection(proj)
    out_geotrans = [min_x, pixel_width, 0, max_y, 0, pixel_height]
    out_ds.SetGeoTransform(out_geotrans)
    inv_geotrans = gdal.InvGeoTransform(out_geotrans)

    for in_fn in tqdm(file_list, desc="Mosaicking Images"):
        in_ds = gdal.Open(in_fn)
        in_gt = in_ds.GetGeoTransform()
        x_off, y_off = map(int, gdal.ApplyGeoTransform(inv_geotrans, in_gt[0], in_gt[3]))
        for b in range(1, band_count + 1):
            data = in_ds.GetRasterBand(b).ReadAsArray()
            out_ds.GetRasterBand(b).WriteArray(data, x_off, y_off)
        in_ds = None

    out_ds.FlushCache()
    out_ds = None
    ref_ds = None
    print(f"[SUCCESS] Mosaic TIFF saved to: {outpath}")


def tif_to_png(tif_path, png_path):
    """将 GeoTIFF 转换为 PNG 预览图"""
    try:
        ds = gdal.Open(tif_path)
        if ds is None:
            print(f"[ERROR] Cannot open TIFF file: {tif_path}")
            return

        band_count = ds.RasterCount
        print(f"[INFO] TIFF bands: {band_count}, Size: {ds.RasterXSize}x{ds.RasterYSize}")

        if band_count >= 3:
            # 多波段（RGB）
            b1 = ds.GetRasterBand(1).ReadAsArray()
            b2 = ds.GetRasterBand(2).ReadAsArray()
            b3 = ds.GetRasterBand(3).ReadAsArray()

            # 确保数据在 0-255 范围内
            b1 = np.clip(b1, 0, 255).astype(np.uint8)
            b2 = np.clip(b2, 0, 255).astype(np.uint8)
            b3 = np.clip(b3, 0, 255).astype(np.uint8)

            rgb = np.dstack((b1, b2, b3))
        elif band_count == 1:
            # 单波段（灰度或分割掩膜）
            band = ds.GetRasterBand(1).ReadAsArray()
            band = np.clip(band, 0, 255).astype(np.uint8)
            rgb = np.stack([band, band, band], axis=-1)
        else:
            print(f"[ERROR] Unsupported band count: {band_count}")
            ds = None
            return

        img = Image.fromarray(rgb)
        img.save(png_path)
        print(f"[SUCCESS] Preview PNG saved to: {png_path}")
        ds = None
    except Exception as e:
        print(f"[ERROR] Failed to convert TIFF to PNG: {str(e)}")
        import traceback
        traceback.print_exc()
        if 'ds' in locals():
            ds = None
        raise


if __name__ == '__main__':
    # 使用绝对路径（基于当前脚本位置向上两级到项目根目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    image_path = os.path.join(project_root, "data", "test", "3")
    result_path = os.path.join(project_root, "data", "test", "4")

    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Input path: {image_path}")
    print(f"[INFO] Output path: {result_path}")
    print(f"[INFO] Input path exists: {os.path.exists(image_path)}")

    os.makedirs(result_path, exist_ok=True)

    imageList = glob.glob(os.path.join(image_path, "*.tif"))
    print(f"[INFO] Found {len(imageList)} TIFF files to mosaic")

    if len(imageList) > 0:
        print(f"[INFO] Files: {[os.path.basename(f) for f in imageList[:5]]}")

    if len(imageList) == 0:
        print("[ERROR] No TIFF files found in input directory!")
        print(f"[DEBUG] Listing directory contents: {image_path}")
        if os.path.exists(image_path):
            files = os.listdir(image_path)
            print(f"[DEBUG] Directory contains: {files}")
        else:
            print(f"[DEBUG] Directory does not exist!")
        sys.exit(1)

    mosaic_tif = os.path.join(result_path, "mosaic_result.tif")
    mosaic_png = os.path.join(result_path, "mosaic_preview.png")

    RasterMosaic(imageList, mosaic_tif)

    if os.path.exists(mosaic_tif):
        tif_to_png(mosaic_tif, mosaic_png)
    else:
        print("[ERROR] Mosaic TIFF was not created!")
        sys.exit(1)
