
from typing import List
import numpy as np
from torch.utils.data import Dataset
import torch
# Dataset generation (modified for NumPy)
def make_grid(nx=32, ny=32):
    xs = np.linspace(-1.0, 1.0, nx)
    ys = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    coords = np.stack([X.reshape(-1), Y.reshape(-1)], axis=-1)
    return coords, (nx, ny)

def true_modes(coords):
    x = coords[:,0]; y = coords[:,1]
    m1 = np.sin(np.pi * (x+1)/2.0) * np.cos(np.pi * (y+1)/2.0)
    m2 = np.cos(np.pi * (x+1)) * np.sin(np.pi * (y+1))
    m3 = np.sin(2*np.pi * x) * np.sin(2*np.pi * y)
    m4 = np.ones_like(x)*0.5
    W = np.stack([m1, m2, m3, m4], axis=1).astype(np.complex64)
    return W

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
       
def synth_sequence(T, norm_T, resolution, dt, sample_ratio=0.1, sigma=0.1, seed=0, normalize_t=False):  # T reduced for speed
    if T is None:
        T = norm_T
    np.random.seed(seed)
    coords_full, shape = make_grid(resolution[0], resolution[1])
    n = coords_full.shape[0]

    W_full = true_modes(coords_full)
    alpha = np.array([-0.01, -0.05, -0.20, -0.01])  # m4에 약한 감쇠
    omega = np.array([ 2.00,  4.00,  1.00,  0.30])  # m4에 아주 약한 진동
    b = np.array([1.0+0.5j, 0.8-0.3j, 0.7+0.2j, 0.2+0.0j], dtype=np.complex64)  # 상수모드 초기계수 축소

    compensate_frame_time = 0.1/dt # make every experiments to have same sequence. 
    coords_list = []
    y_list = []
    y_true_list = []
    y_true_full_list = []
    k = max(1, int(n * sample_ratio))
    idx = np.random.choice(n, size=k, replace=False) #Assumes observation is accessible only at fixed locations.
    for t in range(T):
        phi = np.exp((alpha + 1j*omega) * t*dt*compensate_frame_time) * b
        I = W_full @ phi

        coords_t = coords_full[idx]
        y_t = I[idx]

        noise = sigma * (np.random.normal(size=y_t.shape) + 1j*np.random.normal(size=y_t.shape))
        y_t_noisy = (y_t + noise).astype(np.complex64)

        coords_list.append(coords_t)
        y_list.append(y_t_noisy)
        y_true_list.append(y_t)
        y_true_full_list.append(I)
    t_list = list(range(T))
    if normalize_t:
        t_list = [np.float32(t) / np.float32(norm_T) for t in t_list]
    else:
        t_list = [np.float32(t) * np.float32(dt) for t in t_list]
    # print(f"t_list dtype: {type(t_list[0])}, coords dtype: {coords_list[0].dtype}, y dtype: {y_list[0].dtype}")
    return t_list, coords_list, y_list, y_true_full_list, coords_full, y_true_list,(alpha, omega, b), W_full



def load_synth(device: torch.device, T, norm_T, resolution,sample_ratio, sigma, dt, seed, normalize_t):
    """Loads synthetic sequence and converts to torch (real/imag split).
    Returns:
        t_list: list[float]
        coords_list: list[Tensor[m,2]]
        y_list: list[Tensor[m,2]] (complex split)
        y_true_list, y_true_full_list, coords_full_t: analogous full-res
        gt_params, W_full: passthroughs from synth_sequence
    """
    (
        t_list,
        coords_list,
        y_list,
        y_true_full_list,
        coords_full,
        y_true_list,
        gt_params,
        W_full,
    ) = synth_sequence(T=T, norm_T=norm_T, resolution=resolution, sample_ratio=sample_ratio, sigma=sigma, dt=dt, seed=seed, normalize_t=normalize_t)
        
    def to_torch_split(lst: List[np.ndarray]):
        yr = [torch.from_numpy(np.real(y)).float().to(device) for y in lst]
        yi = [torch.from_numpy(np.imag(y)).float().to(device) for y in lst]
        return [torch.stack([r, i], dim=-1) for r, i in zip(yr, yi)]

    coords_torch = [torch.from_numpy(c).float().to(device) for c in coords_list]
    y_torch = to_torch_split(y_list)
    y_true_torch = to_torch_split(y_true_list)
    y_true_full_torch = to_torch_split(y_true_full_list)
    coords_full_torch = torch.from_numpy(coords_full).float().to(device)

    return (
        t_list,
        coords_torch,
        y_torch,
        y_true_full_torch,
        coords_full_torch,
        y_true_torch,
        gt_params,
        W_full,
    )



class SynthDataset(Dataset):
    def __init__(self, t_list, coords_list, y_list):
        """
        Custom dataset for synthetic sequence data.
        
        Args:
            t_list: List of time steps (float or int)
            coords_list: List of coordinate tensors [m, 2]
            y_list: List of observation tensors [m, 2] (real, imag)
        """
        self.t_list = t_list
        self.coords_list = coords_list
        self.y_list = y_list
        self.length = len(t_list) - 1  # Number of (t_prev, t_next) pairs

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns data for one time step transition.
        
        Returns:
            t_prev: Float, previous time step
            t_next: Float, next time step
            coords: Tensor[m, 2], coordinates at t_next
            y_prev: Tensor[m, 2], observation at t_prev
            y_next: Tensor[m, 2], observation at t_next
        """
        t_prev = (self.t_list[idx])
        t_next = (self.t_list[idx + 1])
        coords = self.coords_list[idx + 1]
        y_prev = self.y_list[idx]
        y_next = self.y_list[idx + 1]
        return t_prev, t_next, coords, y_next, y_prev

class SynthDataset_seq(Dataset):
    def __init__(self, t_list, coords_list, y_list, K):
        """
        Custom dataset for synthetic sequence data.
        
        Args:
            t_list: List of time steps (float or int)
            coords_list: List of coordinate tensors [m, 2]
            y_list: List of observation tensors [m, 2] (real, imag)
        """
        self.t_list = t_list
        self.coords_list = coords_list
        self.y_list = y_list
        self.length = len(t_list) - 1 - K  # Number of (t_prev, t_next) pairs
        self.K = K # sequential data that I'm loading in single step 
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns data for one time step transition.
        
        Returns:
            t_prev: Float, previous time step
            t_next: Float, next time step
            coords: Tensor[m, 2], coordinates at t_next
            y_prev: Tensor[m, 2], observation at t_prev
            y_next: Tensor[m, 2], observation at t_next
        """
        t_prev = (self.t_list[idx])
        t_next = self.t_list[idx + 1:idx + 1+self.K]
        coords = self.coords_list[idx + 1:idx + 1+self.K]
        y_prev = self.y_list[idx: idx+self.K]
        y_next = self.y_list[idx + 1: idx + 1+self.K]
        return t_prev,t_next, coords, y_next, y_prev 