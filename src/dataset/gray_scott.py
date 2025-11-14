
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
    return coords, X,Y, (nx, ny)

def gray_scott_step(u, v, Du, Dv, F, k, dt, dx):
    N = u.shape[0]
    # Laplacian with periodic BC
    lap_u = (np.roll(u, 1, 0) + np.roll(u, -1, 0) + np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4*u) / dx**2
    lap_v = (np.roll(v, 1, 0) + np.roll(v, -1, 0) + np.roll(v, 1, 1) + np.roll(v, -1, 1) - 4*v) / dx**2
    
    uv2 = u * v**2
    du = Du * lap_u - uv2 + F * (1 - u)
    dv = Dv * lap_v + uv2 - (F + k) * v
    
    u_new = u + dt * du
    v_new = v + dt * dv
    
    return np.clip(u_new, 0, 1), np.clip(v_new, 0, 1)


def synth_sequence(
    T,
    norm_T,
    resolution,
    sample_ratio,
    sigma,
    dt,
    seed,
    normalize_t=False,
    save_gif=False,
):
    '''
    0~4999, 0.1 time step -> 50000 steps actually... reduce to 200 steps : save every 25 steps? (save every 2.5 sec. )
    0~2499 -> 100 step 
    2500~4999 -> 100 step for eval 
    '''
    # scale_time = 50
    scale_time = 25

    if T is None:
        T = norm_T
    np.random.seed(seed)
    coords_full, x,y, shape = make_grid(resolution[0], resolution[1])

    dx = 1 / (resolution[0] - 1)

    # Wave-like initial condition
    amplitude = 0.1  # Perturbation amplitude
    u_init = 0.9 + amplitude * np.sin(4 * np.pi * x) * np.cos(2 * np.pi * y)  # Wave pattern
    v_init = 0.1 + 0.05 * np.sin(np.pi * x)  # Lower initial v with simpler wave

    # Parameters for Gray-Scott
    Du, Dv, F, k = 2e-4, 1e-5, 0.035, 0.065

    # Generate multiple trajectories
    F_pert = F + np.random.normal(0, sigma)  # Uncontrollable perturbation
    u = u_init.copy()
    v = v_init.copy()

    


    # traj_u = [u.copy()]
    n = coords_full.shape[0]
    idx = np.random.choice(n, size=int(n*sample_ratio), replace=False)
    y_list = []
    y_list_full = []
    coords_list = []
    frames = []

    for step in range(T*scale_time):
        u, v = gray_scott_step(u, v, Du, Dv, F_pert, k, dt, dx)
        # traj_u.append(u.copy())
        if step % scale_time == 0:
            v_raw = v.copy().reshape(-1)
            y_list_full.append(v_raw)
            y_list.append(v_raw[idx])
            coords_list.append(coords_full[idx])
            if save_gif:
                
                import matplotlib.pyplot as plt
                import imageio
                plt.figure(figsize=(10, 5))
                # plt.imshow(traj_u[step], cmap='viridis', vmin=0, vmax=1)
                plt.scatter(coords_full[idx, 0], coords_full[idx, 1], c=v_raw[idx], cmap='viridis', vmin=0, vmax=1)

                plt.title(f"Time Step {step}")
                plt.xlabel('x')
                plt.ylabel('y')
                plt.colorbar()
                plt.axis('square')
                plt.savefig('temp_frame.png')
                frames.append(imageio.imread('temp_frame.png'))
                plt.close()

    if save_gif:
        imageio.mimsave("./results/gray_scott_fast.gif", frames, fps=10)  # fps 조정 가능
        print("Saved GIF to ./results/gray_scott_fast.gif")
        import os
        # 임시 파일 삭제
        if os.path.exists('temp_frame.png'):
            os.remove('temp_frame.png')
    t_list = list(range(T))
    if normalize_t:
        t_list = [np.float32(t) / np.float32(norm_T) for t in t_list]
    else:
        t_list = [np.float32(t) * np.float32(dt) for t in t_list]
    # y_list = y_list[::-1]
    # y_list_full = y_list_full[::-1]
    return t_list, coords_list, y_list, y_list_full, coords_full



# for NDMD execution, comment out below 
def load_synth(device: torch.device, T, norm_T, resolution,sample_ratio, sigma, dt, seed, normalize_t):
    """Loads synthetic sequence and converts to torch (real/imag split).
    Returns:
        t_list: list[float]
        coords_list: list[Tensor[m,2]]
        y_list: list[Tensor[m,2]] (complex split)
        y_list_full: list[Tensor[m,2]] (complex split)
        coords_full_t: Tensor[n,2]
    """
    (
        t_list,
        coords_list,
        y_list,
        y_list_full,
        coords_full
    ) = synth_sequence(T=T, norm_T=norm_T, resolution=resolution, sample_ratio=sample_ratio, sigma=sigma, dt=dt, seed=seed, normalize_t=normalize_t)
        
    
    def to_torch_split(lst: List[np.ndarray]):
        yr = [torch.from_numpy(np.real(y)).float().to(device) for y in lst]
        yi = [torch.from_numpy(np.imag(y)).float().to(device) for y in lst]
        return [torch.stack([r, i], dim=-1) for r, i in zip(yr, yi)]

    coords_torch = [torch.from_numpy(c).float().to(device) for c in coords_list]
    y_torch = to_torch_split(y_list)
    y_true_full_torch = to_torch_split(y_list_full)
    coords_full_torch = torch.from_numpy(coords_full).float().to(device)

    return (
        t_list,
        coords_torch,
        y_torch,
        y_true_full_torch,
        coords_full_torch,
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
        t_prev = self.t_list[idx]
        t_next = self.t_list[idx + 1]
        coords = self.coords_list[idx + 1]
        y_prev = self.y_list[idx]
        y_next = self.y_list[idx + 1]
        return t_prev, t_next, coords, y_next, y_prev
