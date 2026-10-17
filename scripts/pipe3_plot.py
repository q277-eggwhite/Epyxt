import os
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from astropy.io import fits
from astropy.wcs import WCS 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
from astropy.wcs import FITSFixedWarning

warnings.simplefilter('ignore', category=FITSFixedWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.getcwd()  
os.chdir(BASE_DIR)
print(f"\nWorking directory: {BASE_DIR}")

OUT_DIR = "result/07_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# 输出 PNG 文件名中使用的算法缩写（contbin 简写为 cont）
METHOD_TAG = {"acb": "acb", "wvt": "wvt", "contbin": "cont"}

def load_method_from_config():
    with open("config.json", "r") as f:
        return json.load(f)["step1"].get("method", "acb").lower()

def render_map(fits_file, out_name, cmap_obj, vmin, vmax, label, global_mask=None, mask_color='white'):
    file_path = f"result/06_maps/{fits_file}"
    if not os.path.exists(file_path):
        print(f"Cannot find file {file_path}, skipping plot...")
        return
        
    with fits.open(file_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header
    
    wcs = WCS(header)
    if global_mask is not None:
        data = np.where(global_mask | np.isnan(data) | (data <= 0), np.nan, data)
    else:
        data = np.where((data <= 0) | np.isnan(data), np.nan, data)

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(1, 1, 1, projection=wcs)
    
    # 收紧四周留白：画布仍为 9x9 英寸，输出仍固定 2700x2700 px
    fig.subplots_adjust(left=0.125, right=0.985, bottom=0.11, top=0.985)
    
    # 🎯 优化刻度排版：自动转换为标准的天文时分秒、度分秒坐标刻度
    try:
        ax.coords[0].set_major_formatter('hh:mm:ss')
        ax.coords[1].set_major_formatter('dd:mm:ss')
    except Exception:
        pass

    if isinstance(cmap_obj, str):
        cmap = copy.copy(plt.colormaps[cmap_obj])
    else:
        cmap = copy.copy(cmap_obj)

    im = ax.imshow(data, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax, interpolation='none')

    # 在 mask 区域叠加纯色层（比 cmap.set_bad 更可靠）
    if global_mask is not None:
        mask_rgba = np.zeros((*data.shape, 4))
        mask_rgba[global_mask] = mcolors.to_rgba(mask_color)
        ax.imshow(mask_rgba, origin='lower', interpolation='none')

    ax.set_xlabel('RA', fontsize=14, fontweight='bold', labelpad=5)
    ax.set_ylabel('Dec', fontsize=14, fontweight='bold', labelpad=-1)
    ax.tick_params(axis='both', which='major', labelsize=12)

    valid_y, valid_x = np.where(~np.isnan(data))
    if len(valid_x) > 0 and len(valid_y) > 0:
        pad = 6
        ax.set_xlim(valid_x.min() - pad, valid_x.max() + pad)
        ax.set_ylim(valid_y.min() - pad, valid_y.max() + pad)

    cax = make_axes_locatable(ax).append_axes("bottom", size="4%", pad=0.3, axes_class=plt.Axes)
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=12) 
    cbar.set_label(label, fontsize=14, fontweight='bold', labelpad=6)

    pixel_scale_deg = abs(header.get('CDELT2', 2.4/3600.0))
    five_arcmin_px = (5.0 / 60.0) / pixel_scale_deg

    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()
    bar_x = x_lims[1] - five_arcmin_px - (x_lims[1] - x_lims[0]) * 0.05
    bar_y = y_lims[1] - (y_lims[1] - y_lims[0]) * 0.05
    
    ax.plot([bar_x, bar_x + five_arcmin_px], [bar_y, bar_y], color='black', lw=4)
    ax.text(bar_x + five_arcmin_px / 2.0, bar_y + (y_lims[1] - y_lims[0]) * 0.015, 
            "5 arcmin", color='black', fontsize=14, ha='center', va='bottom', fontweight='bold')

    out_path = os.path.join(OUT_DIR, out_name)
    plt.savefig(out_path, dpi=300, transparent=False)
    plt.close()
    print(f"Saved successfully: {out_path} (Range: {vmin} ~ {vmax})")

def main():
    method = load_method_from_config()
    print(f"Generating smooth scientific plots based on algorithm [{method}]...")
    
    temp_file = f"result/06_maps/{method}_map_temperature.fits"
    err_file = f"result/06_maps/{method}_map_temp_error.fits"
    
    with open("config.json", "r") as f:
        mask_cfg = json.load(f).get("plot_mask", {})
    kt_min = mask_cfg.get("kt_min", 0.1)
    kt_max = mask_cfg.get("kt_max", 20.0)
    max_err_kev = mask_cfg.get("max_err_kev", 10.0)
    mask_color = mask_cfg.get("mask_color", "white")

    if os.path.exists(temp_file) and os.path.exists(err_file):
        temp_data = fits.getdata(temp_file)
        err_data = fits.getdata(err_file)
        global_invalid_mask = (
            (temp_data < kt_min) | 
            (temp_data > kt_max) | 
            (err_data > max_err_kev) | 
            np.isnan(temp_data) | 
            np.isnan(err_data)
        )
    elif os.path.exists(temp_file):
        temp_data = fits.getdata(temp_file)
        global_invalid_mask = (temp_data < kt_min) | (temp_data > kt_max) | np.isnan(temp_data)
    else:
        global_invalid_mask = None

    UNIFIED_CMAP = plt.colormaps['inferno']
    
    tag = METHOD_TAG.get(method, method)
    
    render_map(f"{method}_map_entropy.fits", f"2d_entropy_{tag}.png", UNIFIED_CMAP, 50.0, 300.0, "Pseudo-Entropy (Arb. Units)", global_mask=global_invalid_mask, mask_color=mask_color)
    render_map(f"{method}_map_pressure.fits", f"2d_press_{tag}.png", UNIFIED_CMAP, 0.0, 0.25, "Pseudo-Pressure (Arb. Units)", global_mask=global_invalid_mask, mask_color=mask_color)
    render_map(f"{method}_map_temp_error.fits", f"2d_temp_error_{tag}.png", UNIFIED_CMAP, 0.0, 3.0, "Temperature Error (keV)", global_mask=global_invalid_mask, mask_color=mask_color)
    render_map(f"{method}_map_temperature.fits", f"2d_temp_{tag}.png", UNIFIED_CMAP, 2.0, 9.0, "Temperature (keV)", global_mask=global_invalid_mask, mask_color=mask_color)
    
    print(f"Plotting with bicubic interpolation finished!")

if __name__ == '__main__':
    main()