########################################################################################################################
# 图像裁剪脚本
########################################################################################################################
import time
import os
import sys
import numpy as np
from osgeo import gdal

# 设置标准输出编码为 UTF-8
if sys.platform == 'win32':
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


class GRID:

    def load_image(self, filename):
        image = gdal.Open(filename)

        img_width = image.RasterXSize
        img_height = image.RasterYSize

        img_geotrans = image.GetGeoTransform()
        img_proj = image.GetProjection()
        img_data = image.ReadAsArray(0, 0, img_width, img_height)

        del image

        return img_proj, img_geotrans, img_data

    def write_image(self, filename, img_proj, img_geotrans, img_data):

        if 'int8' in img_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in img_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        if len(img_data.shape) == 3:
            img_bands, img_height, img_width = img_data.shape
        else:
            img_bands, (img_height, img_width) = 1, img_data.shape

        driver = gdal.GetDriverByName('GTiff')
        image = driver.Create(filename, img_width, img_height, img_bands, datatype)

        image.SetGeoTransform(img_geotrans)
        image.SetProjection(img_proj)

        if img_bands == 1:
            image.GetRasterBand(1).WriteArray(img_data)
        else:
            for i in range(img_bands):
                image.GetRasterBand(i + 1).WriteArray(img_data[i])

        del image


if __name__ == '__main__':
    # 使用相对路径（基于当前脚本位置）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    path_img = os.path.join(project_root, "data", "test", "0", "test.tif")
    path_out = os.path.join(project_root, "data", "test", "1")

    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Input image: {path_img}")
    print(f"[INFO] Output directory: {path_out}")

    # 检查输入文件
    if not os.path.exists(path_img):
        print(f"[ERROR] Input image does not exist: {path_img}")
        sys.exit(1)

    # 确保输出目录存在并清空
    if os.path.exists(path_out):
        import shutil

        shutil.rmtree(path_out)
    os.makedirs(path_out, exist_ok=True)

    t_start = time.time()

    run = GRID()
    proj, geotrans, data = run.load_image(path_img)

    channel, height, width = data.shape

    print(f"[INFO] Image size: {width}x{height}, Channels: {channel}")

    patch_size_w = 256
    patch_size_h = 256

    num = 0  # 图像名字序号

    for i in range(height // patch_size_h):
        for j in range(width // patch_size_w):
            num += 1

            sub_image = data[:, i * patch_size_h:(i + 1) * patch_size_h, j * patch_size_w:(j + 1) * patch_size_w]

            px = geotrans[0] + j * patch_size_w * geotrans[1] + i * patch_size_h * geotrans[2]
            py = geotrans[3] + j * patch_size_w * geotrans[4] + i * patch_size_h * geotrans[5]
            new_geotrans = [px, geotrans[1], geotrans[2], py, geotrans[4], geotrans[5]]

            output_path = os.path.join(path_out, f'{num}.tif')
            run.write_image(output_path, proj, new_geotrans, sub_image)
            time_end = time.time()
            print(f'[INFO] Patch {num} saved, time: {round((time_end - t_start), 4)}s')

    t_end = time.time()
    print(f'[SUCCESS] All patches created: {num} patches, total time: {round((t_end - t_start), 4)}s')
