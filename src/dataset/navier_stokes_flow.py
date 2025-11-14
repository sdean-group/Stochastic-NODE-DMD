import os
import math
import torch
import torch.fft as fft
import numpy as np
import xarray as xr
import imageio.v2 as imageio
import matplotlib.pyplot as plt
# --- (파일 상단 유틸 아래에 추가) ---

# -------------------------------------------------------
# 1️⃣ Utility Functions
# -------------------------------------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def make_coords_full_from_linspace(x_vals, y_vals):
    X, Y = np.meshgrid(x_vals, y_vals, indexing="xy")
    coords_full = np.stack([X.reshape(-1), Y.reshape(-1)], axis=-1)
    return coords_full

def to_torch_split_real_only(lst, device):
    yr = [torch.from_numpy(np.array(y, dtype=np.float32)).to(device) for y in lst]
    yi = [torch.zeros_like(r) for r in yr]
    return [torch.stack([r, i], dim=-1) for r, i in zip(yr, yi)]

def load_synth(
    nc_path: str,
    sample_ratio: float = 0.2,
    normalize_t: bool = False,
    device: torch.device = torch.device("cpu"),
    seed: int = 0,
    data_len: int = None
):
    np.random.seed(seed)
    ds = xr.open_dataset(nc_path)
    print(ds.keys())
    vort = ds["vorticity"].transpose("time", "x", "y").values  # (T, n, n)
    tvals = ds["time"].values
    xvals = ds["x"].values
    yvals = ds["y"].values
    coords_full = make_coords_full_from_linspace(xvals, yvals)
    n = coords_full.shape[0]
    m = int(n * sample_ratio)
    idx = np.random.choice(n, size=m, replace=False)

    y_list, y_list_full, coords_list = [], [], []
    for k in range(vort.shape[0]):
        flat = vort[k].reshape(-1)
        y_list_full.append(flat.copy())
        y_list.append(flat[idx])
        coords_list.append(coords_full[idx])

    # Normalize time
    if normalize_t:
        t_list = [np.float32(t / tvals[-1]) for t in tvals]
    else:
        t_list = [np.float32(t) for t in tvals]

    coords_torch = [torch.from_numpy(c).float().to(device) for c in coords_list]
    coords_full_torch = torch.from_numpy(coords_full).float().to(device)
    y_torch = to_torch_split_real_only(y_list, device)
    y_full_torch = to_torch_split_real_only(y_list_full, device)
    # return t_list[60:75], coords_torch[60:75], y_torch[60:75], y_full_torch[60:75], coords_full_torch
    return t_list[:data_len], coords_torch[:data_len], y_torch[:data_len], y_full_torch[:data_len], coords_full_torch

# -------------------------------------------------------
# 4️⃣ Dataset Class
# -------------------------------------------------------
class SynthDataset(torch.utils.data.Dataset):
    def __init__(self, t_list, coords_list, y_list):
        self.t_list = t_list
        self.coords_list = coords_list
        self.y_list = y_list
        # self.t_list = t_list[10:]
        # self.coords_list = coords_list[10:]
        # self.y_list = y_list[10:]
        self.length = len(self.t_list) - 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        t_prev = self.t_list[idx]
        t_next = self.t_list[idx + 1]
        coords = self.coords_list[idx + 1]
        y_prev = self.y_list[idx]
        y_next = self.y_list[idx + 1]
        return t_prev, t_next, coords, y_next, y_prev

# -------------------------------------------------------
# 5️⃣ Example Usage
# -------------------------------------------------------
if __name__ == "__main__":
    
    data_path = "/home/yk826/pojects/torch-cfd/navier_stokes_flow/spectral_vorticity.nc"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_list, coords_list, y_list, y_full, coords_full = load_synth(
        data_path, sample_ratio=0.1, normalize_t=True, device=device
    )
    print(t_list)
    dataset = SynthDataset(t_list, coords_list, y_list)
    print(f"Dataset length: {len(dataset)}")
    tp, tn, c, yn, yp = dataset[0]
    print(f"Example shapes -> coords:{c.shape}, y_next:{yn.shape}, y_prev:{yp.shape}")