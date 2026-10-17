import os
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# ---------------------------------------------------------
# Auto path adaptation
# ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
MAP_DIR = os.path.join(BASE_DIR, "result/06_maps")
OUT_DIR = os.path.join(BASE_DIR, "result/07_plots")

os.makedirs(OUT_DIR, exist_ok=True)

def render_map(fits_file, out_name, cmap_obj, vmin, vmax, label, mask, mask_color='white'):
    file_path = os.path.join(MAP_DIR, fits_file)
    if not os.path.exists(file_path):
        return
        
    data = fits.getdata(file_path)
    
    # Apply global mask based on physical image generation
    plot_data = np.where(mask | np.isnan(data), np.nan, data)

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = copy.copy(cmap_obj)
    
    im = ax.imshow(plot_data, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax, interpolation='none')

    # 在 mask 区域叠加纯色层（比 cmap.set_bad 更可靠）
    mask_rgba = np.zeros((*plot_data.shape, 4))
    mask_rgba[mask] = mcolors.to_rgba(mask_color)
    ax.imshow(mask_rgba, origin='lower', interpolation='none')

    ax.axis('off')

    # Auto crop whitespace
    valid_y, valid_x = np.where(~np.isnan(plot_data))
    if len(valid_x) > 0:
        pad = 15
        ax.set_xlim(valid_x.min() - pad, valid_x.max() + pad)
        ax.set_ylim(valid_y.min() - pad, valid_y.max() + pad)

    # Add Colorbar
    cax = make_axes_locatable(ax).append_axes("bottom", size="4%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(label, fontsize=16, fontweight='bold', labelpad=10)

    # Draw scale bar (5 arcmin)
    bar_x, bar_y = ax.get_xlim()[1] - 31.25 - 15, ax.get_ylim()[1] - 15
    ax.plot([bar_x, bar_x + 31.25], [bar_y, bar_y], color='black', lw=4)
    ax.text(bar_x + 31.25/2, bar_y + 3, "5 arcmin", color='black', fontsize=18, ha='center', va='bottom')

    # Save
    plt.savefig(os.path.join(OUT_DIR, out_name), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_name}")

def main():
    print("Applying hard threshold cleaning on FITS images...")
    
    # Read base layer for mask generation
    temp_p = os.path.join(MAP_DIR, "map_temperature.fits")
    terr_p = os.path.join(MAP_DIR, "map_temp_error.fits")

    if not os.path.exists(temp_p) or not os.path.exists(terr_p):
        print(f"ERROR: Cannot find temperature or error maps in {MAP_DIR}!")
        return

    kT_data = fits.getdata(temp_p)
    kTErr_data = fits.getdata(terr_p)

    config_path = os.path.join(BASE_DIR, "config.json")
    with open(config_path, "r") as f:
        mask_cfg = json.load(f).get("plot_mask", {})
    kt_min = mask_cfg.get("kt_min", 0.1)
    kt_max = mask_cfg.get("kt_max", 20.0)
    max_err_kev = mask_cfg.get("max_err_kev", 10.0)
    mask_color = mask_cfg.get("mask_color", "white")

    # ---------------------------------------------------------
    # Hard logic: Apply absolute threshold directly on image matrix
    # ---------------------------------------------------------
    # 1. Remove temperature outliers: kT < kt_min or kT > kt_max
    # 2. Remove error outliers: kT_err > max_err_kev
    # 3. Remove pixels that are originally NaN
    global_mask = (kT_data < kt_min) | (kT_data > kt_max) | (kTErr_data > max_err_kev) | np.isnan(kT_data)
    
    # ---------------------------------------------------------
    # Custom colormap
    # ---------------------------------------------------------
    custom_afmhot = ListedColormap(plt.colormaps['afmhot'](np.linspace(0, 0.85, 256)))
    custom_redblack = LinearSegmentedColormap.from_list('pure_rb', ['black', 'red'])

    # ---------------------------------------------------------
    # Unified rendering
    # ---------------------------------------------------------
    tasks = [
        ("map_entropy.fits", "2d_entropy.png", plt.colormaps['magma'], 20.0, 100.0, "Pseudo-Entropy"),
        ("map_pressure.fits", "2d_pressure.png", plt.colormaps['plasma'], 0.1, 0.4, "Pseudo-Pressure"),
        ("map_temp_error.fits", "2d_temp_error.png", custom_afmhot, 0.0, 3.0, "Temperature Error (keV)"),
        ("map_temperature.fits", "2d_temperature_afmhot.png", custom_afmhot, 2.0, 9.0, "Temperature (keV)"),
        ("map_temperature.fits", "2d_temperature_redblack.png", custom_redblack, 2.0, 9.0, "Temperature (keV)")
    ]

    for fits_f, out_f, cmap, vmin, vmax, label in tasks:
        render_map(fits_f, out_f, cmap, vmin, vmax, label, global_mask, mask_color=mask_color)

    print(f"\nAll garbage data cleaned. Results in: {OUT_DIR}")

if __name__ == '__main__':
    main()