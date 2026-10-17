import os
import sys
import json
import shutil
import glob
import subprocess
import multiprocessing as mp
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.getcwd()  # Use current working directory as data directory
os.chdir(BASE_DIR)
print(f"\nWorking directory: {BASE_DIR}")

def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

def init_directories():
    result_dir = os.path.join(BASE_DIR, "result")
    print("\n[Step 0] Initializing result directory...")
    
    dirs_to_create = [
        "01_preparation", "01_preparation/regions", 
        "02_extraction", "03_matrices", "04_fitting", 
        "05_fit_plots", "06_maps", "07_plots"
    ]
    for d in dirs_to_create:
        os.makedirs(os.path.join(result_dir, d), exist_ok=True)
    print("Directory structure ready!")

def extract_regions_from_map(bin_map_path, header, prefix):
    from skimage import measure
    print(f"Extracting {prefix} regions (Source: {os.path.basename(bin_map_path)})...")
    
    raw_data = fits.getdata(bin_map_path)   
    data = np.nan_to_num(raw_data, nan=-1).astype(int)
    wcs = WCS(header)
    unique_bins = np.unique(data)
    unique_bins = unique_bins[unique_bins >= 0]
    out_dir = "result/01_preparation/regions"
    
    count = 0
    for b in unique_bins:
        mask = (data == b).astype(float)
        padded_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
        contours = measure.find_contours(padded_mask, 0.5)
        
        reg_lines = ["fk5"]
        if contours:
            for contour in contours:
                y_i, x_i = contour[:, 0] - 1, contour[:, 1] - 1
                ra, dec = wcs.pixel_to_world_values(x_i, y_i)
                coord_str = ",".join([f"{r:.6f}d,{d:.6f}d" for r, d in zip(ra, dec)])
                reg_lines.append(f"polygon({coord_str})")
        else:
            yy, xx = np.where(mask == 1)
            for ty, tx in zip(yy, xx):
                corners_x = [tx-0.5, tx+0.5, tx+0.5, tx-0.5]
                corners_y = [ty-0.5, ty-0.5, ty+0.5, ty+0.5]
                ra, dec = wcs.pixel_to_world_values(corners_x, corners_y)
                coord_str = ",".join([f"{r:.6f}d,{d:.6f}d" for r, d in zip(ra, dec)])
                reg_lines.append(f"polygon({coord_str})")

        with open(os.path.join(out_dir, f"{prefix}_bin_{int(b)}.reg"), "w") as f:
            f.write("\n".join(reg_lines) + "\n")
        count += 1
    print(f"Successfully generated {count} {prefix} region files!")

def step1_make_mask(cfg):
    print("\n==========================================")
    print("[Step 1] Generating mask (Calling CIAO dmcopy)")
    print("==========================================")
    raw_file = os.path.join("data", cfg["base"]["base_image"])
    reg_file = os.path.join("data", cfg["step1"]["region_file"])
    holes_image = "result/01_preparation/fxt_counts_holes.fits"
    output_mask = "result/01_preparation/mask_joint.fits"
    c_ra, c_dec = cfg["base"]["center_ra"], cfg["base"]["center_dec"]
    r_arcmin = cfg["base"]["radius_arcmin"]

    cmd = f'dmcopy "{raw_file}[exclude region({reg_file})]" {holes_image} clobber=yes'
    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
    with fits.open(holes_image) as hdul:
        data, header = hdul[0].data, hdul[0].header
    wcs = WCS(header)
    y_idx, x_idx = np.indices(data.shape)
    coords = wcs.pixel_to_world(x_idx, y_idx)
    center_coord = SkyCoord(ra=c_ra*u.deg, dec=c_dec*u.deg, frame='icrs')
    my_mask = (coords.separation(center_coord) <= r_arcmin * u.arcmin).astype(int)
    my_mask[data == 0] = 0
    my_mask[np.isnan(data)] = 0
    fits.PrimaryHDU(my_mask, header=header).writeto(output_mask, overwrite=True)
    print(f"Exclusive mask generated: {output_mask}")

# --- ACB ---
g_counts, g_mask, g_radii, g_target_sn = None, None, None, None
def init_worker(counts, mask, radii, target_sn):
    global g_counts, g_mask, g_radii, g_target_sn
    g_counts, g_mask, g_radii, g_target_sn = counts, mask, radii, target_sn

def binary_search_worker(coord):
    y, x = coord
    if g_mask[y, x] == 0: return y, x, 0.0
    ny, nx = g_counts.shape
    left, right, best_r = 0, len(g_radii) - 1, 0.0
    while left <= right:
        mid = (left + right) // 2
        r = g_radii[mid]
        r_int = int(np.ceil(r))
        y_min, y_max = max(0, y - r_int), min(ny, y + r_int + 1)
        x_min, x_max = max(0, x - r_int), min(nx, x + r_int + 1)
        sub_c = g_counts[y_min:y_max, x_min:x_max]
        sub_m = g_mask[y_min:y_max, x_min:x_max]
        yy, xx = np.ogrid[y_min-y:y_max-y, x_min-x:x_max-x]
        valid = ((xx**2 + yy**2) <= r**2) & (sub_m == 1)
        c_tot = np.sum(sub_c[valid])
        if (np.sqrt(c_tot) if c_tot > 0 else 0.0) >= g_target_sn:
            best_r, right = r, mid - 1
        else:
            left = mid + 1
    return y, x, best_r

def run_acb_binning(counts_data, mask_data, header, cfg):
    print("Starting ACB (Adaptive Circular Binning) algorithm...")
    target_sn, max_r = cfg["step1"]["target_sn"], cfg["step1"]["max_radius"]
    step = 1.0 
    search_radii = np.arange(1.0, max_r + step, step)
    coords = list(zip(*np.where(mask_data == 1)))
    scale_map = np.zeros_like(counts_data, dtype=float)
    with mp.Pool(max(1, mp.cpu_count() - 2), initializer=init_worker, initargs=(counts_data, mask_data, search_radii, target_sn)) as pool:
        for y, x, r in pool.imap_unordered(binary_search_worker, coords, chunksize=50):
            scale_map[y, x] = r
    fits.PrimaryHDU(scale_map, header=header).writeto("result/01_preparation/acb_scale_map.fits", overwrite=True)
    print("ACB binning map saved!")

# --- WVT ---
def run_wvt_binning(counts_data, mask_data, header, cfg):
    print("Starting Signal-Weighted CVT binning algorithm...")
    from collections import deque
    from scipy.spatial import cKDTree

    target_snr = cfg["step1"]["target_sn"]
    rows, cols = counts_data.shape
    snr_map = np.sqrt(np.maximum(counts_data, 0))
    mask = (mask_data == 1)
    bin_map = -1 * np.ones((rows, cols), dtype=int)
    unbinned = mask.copy()
    coords = np.argwhere(unbinned)
    snr_values = snr_map[unbinned]
    order = np.argsort(-snr_values)
    coords = coords[order]
    bin_list, centroids = [], []

    print(f"Phase 1/2: BFS region growing (Target S/N = {target_snr})...")
    for (i, j) in coords:
        if not unbinned[i, j]: continue
        current_bin = [(i, j)]
        unbinned[i, j] = False
        bin_snr2 = snr_map[i, j] ** 2
        q = deque([(i, j)])
        while q and np.sqrt(bin_snr2) < target_snr:
            pi, pj = q.popleft()
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0: continue
                    ni, nj = pi + di, pj + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if unbinned[ni, nj]:
                            current_bin.append((ni, nj))
                            unbinned[ni, nj] = False
                            bin_snr2 += snr_map[ni, nj] ** 2
                            q.append((ni, nj))
        if np.sqrt(bin_snr2) >= target_snr:
            bin_list.append(current_bin)
            centroids.append(np.mean(np.array(current_bin), axis=0))

    centroids = np.array(centroids)
    if len(centroids) == 0:
        print("Error: Failed to generate any regions meeting the S/N requirements!")
        return

    print(f"Phase 2/2: Lloyd iteration to optimize shape ({len(centroids)} initial centroids)...")
    valid_coords = np.argwhere(mask)
    current_centroids = centroids.copy()
    for it in range(15):
        tree = cKDTree(current_centroids)
        _, labels = tree.query(valid_coords)
        new_centroids = []
        for idx in range(current_centroids.shape[0]):
            points = valid_coords[labels == idx]
            if points.shape[0] == 0:
                new_centroids.append(current_centroids[idx])
            else:
                weights = snr_map[points[:, 0], points[:, 1]]
                new_centroids.append(np.average(points, axis=0, weights=weights) if np.sum(weights) > 0 else np.mean(points, axis=0))
        new_centroids = np.array(new_centroids)
        if np.all(np.linalg.norm(new_centroids - current_centroids, axis=1) < 0.1):
            print(f"Centroids converged after {it+1} iterations!")
            break
        current_centroids = new_centroids

    tree = cKDTree(current_centroids)
    _, final_labels = tree.query(valid_coords)
    for idx, (i, j) in enumerate(valid_coords): bin_map[i, j] = final_labels[idx]

    bin_map_fits = "result/01_preparation/wvt_native_bins.fits"
    fits.PrimaryHDU(bin_map, header=header).writeto(bin_map_fits, overwrite=True)
    extract_regions_from_map(bin_map_fits, header, "wvt")

# --- Contbin ---
def run_contbin(counts_data, mask_data, header, cfg):
    print("Starting official C++ native Contbin algorithm (without background)...")
    prep_dir = os.path.join(os.getcwd(), "result", "01_preparation")
    contbin_exe = "contbin"
    
    input_image = os.path.join(prep_dir, "fxt_counts_holes.fits")
    mask_file = os.path.join(prep_dir, "mask_joint.fits")
    out_binmap = os.path.join(prep_dir, "contbin_native_bins.fits")
    target_sn = cfg["step1"].get("target_sn", 50.0)

    cmd = [
        contbin_exe, 
        f"--sn={target_sn}", 
        f"--mask={mask_file}", 
        "--constrainfill", 
        "--constrainval=2.0", 
        f"--outbinmap={out_binmap}", 
        input_image
    ]
    cmd_str = " ".join(cmd)
    try:
        subprocess.run(cmd_str, shell=True, check=True)
        print(f"Contbin ran successfully! Map saved to: {out_binmap}")
        extract_regions_from_map(out_binmap, header, "contbin")
    except subprocess.CalledProcessError as e:
        print(f"Execution failed with return code: {e.returncode}")

def step2_spatial_binning(cfg):
    print("\n==========================================")
    print("[Step 2] Spatial Binning Calculation")
    print("==========================================")
    raw_file = os.path.join("data", cfg["base"]["base_image"])
    mask_file = "result/01_preparation/mask_joint.fits"
    counts_data, header = fits.getdata(raw_file, header=True)
    mask_data = fits.getdata(mask_file)
    method = cfg["step1"].get("method", "acb").lower()
    
    # --- Strict parameter whitelist validation ---
    valid_methods = ["acb", "wvt", "contbin"]
    if method not in valid_methods:
        raise ValueError(
            f"Invalid spatial binning method specified in config: '{method}'. "
            f"Allowed methods are: {valid_methods}."
        )
    # -----------------------------------------------
    
    if method == "wvt": 
        run_wvt_binning(counts_data, mask_data, header, cfg)
    elif method == "contbin": 
        run_contbin(counts_data, mask_data, header, cfg)
    else: 
        run_acb_binning(counts_data, mask_data, header, cfg)

if __name__ == '__main__':
    init_directories()
    config = load_config()
    step1_make_mask(config)
    step2_spatial_binning(config)
    print("\nPhase 1 tasks completed successfully!")