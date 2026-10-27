import os
import sys
import json
import glob
import subprocess
import multiprocessing as mp
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
import re
import tempfile   
import warnings
from astropy.wcs import FITSFixedWarning
from scipy.interpolate import griddata

warnings.simplefilter('ignore', category=FITSFixedWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.getcwd()  
os.chdir(BASE_DIR)
print(f"\nWorking directory: {BASE_DIR}")

GLOBAL_CFG = None
GLOBAL_OBS_LIST = None  

def load_config():
    with open("config.json", "r") as f:
        return json.load(f) 

def auto_load_observations(data_dir):
    obs_list = []
    if not os.path.exists(data_dir):
        print(f"ERROR: Cannot find data directory {data_dir}")
        return obs_list
        
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and re.match(r'^\d{11}$', item):
            obs_id = item
            obs_dict = {
                "obs_id": obs_id,
                "bkg_a": os.path.abspath(os.path.join(item_path, "bkg_a.pha")),
                "bkg_b": os.path.abspath(os.path.join(item_path, "bkg_b.pha"))
            }
            for file in os.listdir(item_path):
                if '_a_' in file and file.endswith('.fits'):
                    obs_dict['evt_a'] = os.path.join(item_path, file)
                elif '_b_' in file and file.endswith('.fits'):
                    obs_dict['evt_b'] = os.path.join(item_path, file)
                elif '_a_' in file and file.endswith('.expo'):
                    obs_dict['exp_a'] = os.path.abspath(os.path.join(item_path, file))
                elif '_b_' in file and file.endswith('.expo'):
                    obs_dict['exp_b'] = os.path.abspath(os.path.join(item_path, file))
            
            required_keys = ['evt_a', 'evt_b', 'exp_a', 'exp_b', 'bkg_a', 'bkg_b']
            if all(key in obs_dict for key in required_keys):
                obs_list.append(obs_dict)
                
    return obs_list

def extract_pha_xselect(evt_file, out_pha, reg_file, region_id, mod, obs_id=""):
    evt_file_abs = os.path.abspath(evt_file)
    out_pha_abs = os.path.abspath(out_pha)
    reg_file_abs = os.path.abspath(reg_file)

    if os.path.exists(out_pha_abs): 
        os.remove(out_pha_abs)
        
    unique_session = f"s_{region_id}_{obs_id}_{mod}"
    evt_dir = os.path.dirname(evt_file_abs)
    evt_name = os.path.basename(evt_file_abs)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_env = os.environ.copy()
        sys_pfiles = custom_env.get('PFILES', '').split(';')[-1] 
        custom_pfiles_dir = os.path.join(tmpdir, "pfiles")
        os.makedirs(custom_pfiles_dir, exist_ok=True)
        custom_env['PFILES'] = f"{custom_pfiles_dir};{sys_pfiles}"
        
        cmds = f"{unique_session}\nread events {evt_name}\n{evt_dir}\nyes\nfilter region {reg_file_abs}\nextract spectrum\nsave spectrum {out_pha_abs}\nexit\nno\n"
        subprocess.run("xselect", input=cmds, text=True, capture_output=True, shell=True, cwd=tmpdir, env=custom_env)
        
    return os.path.exists(out_pha_abs)

def process_single_extraction(task_data):
    region_id, reg_file, method = task_data
    all_success = True
    for obs in GLOBAL_OBS_LIST:
        obs_id = obs["obs_id"]
        ok_a = extract_pha_xselect(obs["evt_a"], f"result/02_extraction/{method}_reg_{region_id}_{obs_id}_A.pha", reg_file, region_id, "A", obs_id)
        ok_b = extract_pha_xselect(obs["evt_b"], f"result/02_extraction/{method}_reg_{region_id}_{obs_id}_B.pha", reg_file, region_id, "B", obs_id)
        if not (ok_a and ok_b): 
            all_success = False
    return (region_id, all_success)

def process_single_fitting(task_data):
    region_id, method = task_data
    cfg = GLOBAL_CFG
    pi_files = [] 
    generated_files_pool = [] 

    for obs in GLOBAL_OBS_LIST:
        obs_id = obs["obs_id"]
        for mod in ['A', 'B']:
            in_pha  = f"result/02_extraction/{method}_reg_{region_id}_{obs_id}_{mod}.pha"
            out_rmf = f"result/03_matrices/{method}_reg_{region_id}_{obs_id}_{mod}.rmf"
            out_arf = f"result/03_matrices/{method}_reg_{region_id}_{obs_id}_{mod}.arf"
            out_pi  = f"result/03_matrices/{method}_reg_{region_id}_{obs_id}_{mod}.pi"
            bkg_f, exp_f = obs[f"bkg_{mod.lower()}"], obs[f"exp_{mod.lower()}"]
            
            if not os.path.exists(in_pha): continue
            generated_files_pool.extend([in_pha, out_rmf, out_arf, out_pi])
            
            subprocess.run(f"fxtrmfgen specfile={in_pha} outfile={out_rmf} clobber=yes", shell=True, capture_output=True)
            subprocess.run(f"fxtarfgen specfile={in_pha} expfile={exp_f} outfile={out_arf} extend=1 clobber=yes", shell=True, capture_output=True)
            
            if os.path.exists(out_rmf) and os.path.exists(out_arf):
                abs_bkg = os.path.abspath(bkg_f)
                abs_rmf = os.path.abspath(out_rmf)
                abs_arf = os.path.abspath(out_arf)
                
                grp_in = f"group min 30\nchkey BACKFILE {abs_bkg}\nchkey RESPFILE {abs_rmf}\nchkey ANCRFILE {abs_arf}\nexit\n"
                subprocess.run(f"grppha infile={in_pha} outfile={out_pi} clobber=yes", input=grp_in, shell=True, capture_output=True, text=True)
                if os.path.exists(out_pi):
                    pi_files.append(os.path.abspath(out_pi))

    fit_success = False
    log_file = os.path.abspath(f"result/04_fitting/{method}_fit_result_{region_id}.log")
    xcm_file = os.path.abspath(f"result/04_fitting/{method}_fit_{region_id}.xcm")
    chain_file = os.path.abspath(f"result/04_fitting/{method}_chain_{region_id}.fits")
    plot_base = os.path.abspath(f"result/05_fit_plots/{method}_plot_{region_id}.png")  
    
    if len(pi_files) > 0:
        fit = cfg["step4_fit"]
        emin, emax = fit.get("energy_range", [0.7, 8.0])
        fit_method = fit.get("fit_method", "lm").lower()
        
        data_cmds = [f"data {idx}:{idx} {pure_path}" for idx, pure_path in enumerate(pi_files, 1)]
        data_inputs_str = "\n".join(data_cmds)

        if fit_method == "mcmc":
            # MCMC 模式：先 LM 拟合找初始点，再用 chain 命令采样
            xcm_content = (
                f"query yes\n{data_inputs_str}\nign bad\nign **-{emin}\nign {emax}-**\n"
                f"statistic cstat\nmodel tbabs*apec\n"
                f"{fit['nh']} -1\n{fit['kt']} 1 0.1 0.1 0.1 20 20\n{fit['abund']} -1\n{fit['redshift']} -1\n{fit['norm']} 1 0 0 0 1e10 1e10\n"
                f"/*\nrenorm\nchatter 10\nfit 100\n"
                f"chain burn 1000\nchain length 50000\nchain run {chain_file}\n"
                f"log {log_file}\nshow parameters\nlog none\n"
                f"cpd {plot_base}/png\nsetplot energy\nplot ldata res\nexit\n"
            )
        else:
            # 默认 LM 模式
            xcm_content = (
                f"query yes\n{data_inputs_str}\nign bad\nign **-{emin}\nign {emax}-**\n"
                f"statistic cstat\nmodel tbabs*apec\n"
                f"{fit['nh']} -1\n{fit['kt']} 1 0.1 0.1 0.1 20 20\n{fit['abund']} -1\n{fit['redshift']} -1\n{fit['norm']} 1 0 0 0 1e10 1e10\n"
                f"/*\nrenorm\nchatter 10\nlog {log_file}\nfit 100\nerror 2 5\nshow parameters\nlog none\n"
                f"cpd {plot_base}/png\nsetplot energy\nplot ldata res\nexit\n"
            )
        
        with open(xcm_file, 'w') as f: f.write(xcm_content)
            
        try:
            subprocess.run(f"xspec - < {xcm_file}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=BASE_DIR)
            fit_success = os.path.exists(log_file)
            for wp in glob.glob(f"{plot_base}_*"):
                new_name = wp.replace(".png_", "_") + ".png"
                if os.path.exists(new_name): os.remove(new_name)
                os.rename(wp, new_name)
        except Exception: pass
            
        for temp_file in generated_files_pool:
            try:
                if os.path.exists(temp_file): os.remove(temp_file)
            except Exception: pass
        try:
            if os.path.exists(xcm_file): os.remove(xcm_file)
        except Exception: pass

    return region_id, fit_success

def get_acb_grid_coords(scale_map, stride):
    grid_step = max(1, int(stride))
    ny, nx = scale_map.shape
    
    y_base = np.arange(0, ny, grid_step)
    x_base = np.arange(0, nx, grid_step)
    
    valid_coords = []
    # 使用固定的随机数种子，保证每次运行结果稳定可重复
    rng = np.random.default_rng(42)
    
    for y in y_base:
        for x in x_base:
            # 🎯 核心升级 1：在固定的规则网格内加入随机抖动，打破死板的方块边界
            jitter_y = y + rng.integers(0, grid_step)
            jitter_x = x + rng.integers(0, grid_step)
            
            # 确保随机偏移后的坐标不越界
            if jitter_y < ny and jitter_x < nx:
                if scale_map[jitter_y, jitter_x] > 0:
                    valid_coords.append((jitter_y, jitter_x))
                    
    return valid_coords

def run_extraction():
    method = GLOBAL_CFG["step1"].get("method", "acb").lower()
    print(f"\n[Step 3] XSELECT spectrum extraction (Current algorithm: {method})")
    tasks = []
    
    if method in ["wvt", "contbin"]:
        reg_files = glob.glob(f"result/01_preparation/regions/{method}_bin_*.reg")
        for reg in reg_files:
            bin_id = os.path.basename(reg).replace(f"{method}_bin_", "").replace(".reg", "")
            tasks.append((bin_id, reg, method))
    else:
        scale_map = fits.getdata("result/01_preparation/acb_scale_map.fits")
        ref_hdr = fits.getheader(os.path.join("data", GLOBAL_CFG["base"]["base_image"]))
        wcs_sys = WCS(ref_hdr)
        pixel_scale_deg = abs(ref_hdr['CDELT2'])
        
        cfg_stride = GLOBAL_CFG["step2"].get("pixel_stride", 4)
        print(f"Using 2D Grid sampling with pixel stride: {cfg_stride}")
        valid_coords = get_acb_grid_coords(scale_map, cfg_stride)
        
        for y, x in valid_coords:
            r_deg = float(scale_map[y, x]) * pixel_scale_deg
            ra, dec = wcs_sys.pixel_to_world_values(x, y)
            region_id = f"{x}_{y}"
            reg_file = f"result/01_preparation/regions/temp_{region_id}.reg"
            with open(reg_file, "w") as f: f.write(f"fk5\ncircle({ra}d, {dec}d, {r_deg}d)\n")
            tasks.append((region_id, reg_file, method))

    total_tasks = len(tasks)
    print(f"Discovered {total_tasks} regions to extract, starting multi-process extraction...")
    workers = max(1, mp.cpu_count() - 2)
    with mp.Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(process_single_extraction, tasks), 1):
            if i % 10 == 0 or i == total_tasks: print(f"Extraction progress: {i} / {total_tasks}")

def run_fitting():
    method = GLOBAL_CFG["step1"].get("method", "acb").lower()
    print(f"\n[Step 4] Generate response matrix and XSPEC fitting (Current algorithm: {method})")
    
    region_ids = []
    if method in ["wvt", "contbin"]:
        reg_files = glob.glob(f"result/01_preparation/regions/{method}_bin_*.reg")
        for reg in reg_files:
            bin_id = os.path.basename(reg).replace(f"{method}_bin_", "").replace(".reg", "")
            region_ids.append(bin_id)
    else:
        scale_map = fits.getdata("result/01_preparation/acb_scale_map.fits")
        cfg_stride = GLOBAL_CFG["step2"].get("pixel_stride", 4)
        valid_coords = get_acb_grid_coords(scale_map, cfg_stride)
        for y, x in valid_coords:
            region_ids.append(f"{x}_{y}")
            
    total_tasks = len(region_ids)
    tasks = [(rid, method) for rid in region_ids]
    print(f"Preparing to fit {total_tasks} regions...")
    with mp.Pool(max(1, mp.cpu_count() - 2)) as pool:
        for i, _ in enumerate(pool.imap_unordered(process_single_fitting, tasks), 1):
            if i % 10 == 0 or i == total_tasks: print(f"Fitting progress: {i} / {total_tasks}")

def run_mapping():
    method = GLOBAL_CFG["step1"].get("method", "acb").lower()
    print(f"\n[Step 5] Generate thermodynamic physical maps and CSV data tables")
    ref_hdr = fits.getheader(os.path.join("data", GLOBAL_CFG["base"]["base_image"]))
    pixel_scale_deg = abs(ref_hdr.get('CDELT2', 2.4/3600.0))
    pixel_area_sq_arcmin = (pixel_scale_deg * 60.0)**2
    ny, nx = ref_hdr['NAXIS2'], ref_hdr['NAXIS1']
    maps = {k: np.full((ny, nx), np.nan) for k in ['kt', 'err', 'p', 'k']}
    results = []
    
    if method in ["wvt", "contbin"]:
        bin_map_file = f"result/01_preparation/{method}_native_bins.fits"
        bin_map = fits.getdata(bin_map_file) if os.path.exists(bin_map_file) else None
    else:
        scale_map = fits.getdata("result/01_preparation/acb_scale_map.fits")

    fit_method = GLOBAL_CFG["step4_fit"].get("fit_method", "lm").lower()
    logs = glob.glob(f"result/04_fitting/{method}_fit_result_*.log")
    for log_f in logs:
        region_id = os.path.basename(log_f).replace(f"{method}_fit_result_", "").replace(".log", "")
        kt, kt_err, norm = np.nan, np.nan, np.nan
        kt_ci_err = np.nan  # 90% 置信区间误差

        if fit_method == "mcmc":
            # MCMC 模式：从 chain FITS 文件读取后验分布
            chain_f = f"result/04_fitting/{method}_chain_{region_id}.fits"
            if os.path.exists(chain_f):
                try:
                    chain_data = fits.getdata(chain_f)
                    col_names = chain_data.columns.names
                    # XSPEC chain 列名格式: 'kT__2', 'norm__5' (参数名__编号)
                    # 兼容可能的 'p2', 'p5' 格式
                    kt_col = None
                    norm_col = None
                    for c in col_names:
                        if c.startswith('kT') or c == 'p2':
                            kt_col = c
                        elif c.startswith('norm') or c == 'p5':
                            norm_col = c
                    if kt_col is not None:
                        kt_samples = chain_data[kt_col]
                        kt = float(np.median(kt_samples))
                        kt_ci_err = (np.percentile(kt_samples, 95) - np.percentile(kt_samples, 5)) / 2.0
                    if norm_col is not None:
                        norm_samples = chain_data[norm_col]
                        norm = float(np.median(norm_samples))
                except Exception as e:
                    print(f"  Warning: failed to parse chain for {region_id}: {e}")
        else:
            # LM 模式：从日志解析
            with open(log_f, 'r', errors='ignore') as f:
                for line in f:
                    if "apec" in line and "kT" in line and "keV" in line:
                        match_err = re.search(r'keV\s+([0-9\.eE\-\+]+)\s+\+/-\s+([0-9\.eE\-\+]+)', line)
                        if match_err: kt, kt_err = float(match_err.group(1)), float(match_err.group(2))
                        else:
                            match_val = re.search(r'keV\s+([0-9\.eE\-\+]+)', line)
                            if match_val: kt = float(match_val.group(1))
                    if "apec" in line and "norm" in line:
                        match_err = re.search(r'norm\s+([0-9\.Ee\-\+]+)\s+\+/-\s+([0-9\.Ee\-\+]+)', line)
                        if match_err: norm = float(match_err.group(1))
                        else:
                            match_val = re.search(r'norm\s+([0-9\.Ee\-\+]+)', line)
                            if match_val: norm = float(match_val.group(1))
        # LM 模式使用 1σ 协方差误差（参数表中的 +/- 值），不使用 90% CI
        # error 2 5 命令仍会运行，但其输出不用于误差计算

        if not np.isnan(kt) and not np.isnan(norm) and norm > 0:
            if method in ["wvt", "contbin"]:
                b_id = int(region_id)
                area = np.sum(bin_map == b_id) * pixel_area_sq_arcmin if bin_map is not None else 1.0
            else:
                x, y = map(int, region_id.split("_"))
                area = np.pi * (scale_map[y, x] ** 2) * pixel_area_sq_arcmin

            pdens = np.sqrt(norm / area)
            pressure = kt * pdens
            entropy = kt / (pdens ** (2/3))
            
            if method in ["wvt", "contbin"]:
                if bin_map is not None:
                    mask = (bin_map == b_id)
                    maps['kt'][mask], maps['err'][mask] = kt, kt_err
                    maps['p'][mask], maps['k'][mask] = pressure, entropy
                results.append({"Bin_ID": b_id, "Area_arcmin2": area, "kT": kt, "kT_err": kt_err, "norm": norm, "Pressure": pressure, "Entropy": entropy})
            else:
                maps['kt'][y, x], maps['err'][y, x] = kt, kt_err
                maps['p'][y, x], maps['k'][y, x] = pressure, entropy
                results.append({"X": x, "Y": y, "Radius_px": scale_map[y, x], "kT": kt, "kT_err": kt_err, "norm": norm, "Pressure": pressure, "Entropy": entropy})

    if len(results) > 0:
        pd.DataFrame(results).to_csv(f"result/06_maps/{method}_thermodynamic_results.csv", index=False)

    def fill_and_mask(data_map):
        if method in ["wvt", "contbin"]: return data_map
        y_idx, x_idx = np.where(~np.isnan(data_map))
        if len(y_idx) == 0: return data_map
        grid_x, grid_y = np.meshgrid(np.arange(nx), np.arange(ny))
        
        # 🎯 核心升级 2：改用 'nearest'（最近邻插值）重构平滑数据表面，保证物理突变不被平滑融化，保留微小真实颗粒感
        filled = griddata(np.column_stack((x_idx, y_idx)), data_map[y_idx, x_idx], (grid_x, grid_y), method='nearest')
            
        final = np.full((ny, nx), np.nan)
        final[scale_map > 0] = filled[scale_map > 0]
        return final

    fits.PrimaryHDU(fill_and_mask(maps['kt']), header=ref_hdr).writeto(f"result/06_maps/{method}_map_temperature.fits", overwrite=True)
    fits.PrimaryHDU(fill_and_mask(maps['err']), header=ref_hdr).writeto(f"result/06_maps/{method}_map_temp_error.fits", overwrite=True)
    fits.PrimaryHDU(fill_and_mask(maps['p']), header=ref_hdr).writeto(f"result/06_maps/{method}_map_pressure.fits", overwrite=True)
    fits.PrimaryHDU(fill_and_mask(maps['k']), header=ref_hdr).writeto(f"result/06_maps/{method}_map_entropy.fits", overwrite=True)
    print(f"Physical FITS maps based on {method} generated successfully with nearest interpolation!")

if __name__ == '__main__':
    GLOBAL_CFG = load_config()

    if len(sys.argv) > 1 and sys.argv[1] == '--map-only':
        print("[Map-only mode] Skipping extraction and fitting, regenerating maps from existing logs...")
        run_mapping()
    else:
        data_directory = GLOBAL_CFG["base"].get("data_dir", "data")

        GLOBAL_OBS_LIST = auto_load_observations(data_directory)
        print(f"Scan complete! Loaded {len(GLOBAL_OBS_LIST)} valid observation data blocks.")

        if len(GLOBAL_OBS_LIST) == 0:
            print("CRITICAL ERROR: No valid data folders found!")
            sys.exit(1)

        run_extraction()
        run_fitting()
        run_mapping()