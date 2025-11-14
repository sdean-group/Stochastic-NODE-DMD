from dataclasses import dataclass
import torch

@dataclass
class Synthetic_Dataset_Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    r: int = 4
    hidden_dim: int = 64
    num_epochs: int =15_00
    batch_size: int = 1
    lr: float = 5e-4
    kl_phi_weight: float =1e-3
    cons_weight: float = 0.15
    recon_weight: float = 3.
    save_dir: str = "results/synthetic/test_run"
    print_every: int = 500
    ode_steps: int = 5
    process_noise: float = 1e-5
    cov_eps: float = 1e-6
    seed: int = 42
    data_len: int = 50
    eval_data_len: int = 100
    sample_ratio: float = 0.1
    sigma: float = 0.001
    dt: float = 0.1  # dt for data generation 
    resolution: tuple = (32,32)  # Added resolution parameter
    train_mode: str = "teacher_forcing"  # "teacher_forcing" or "autoreg" or "evolve"
    normalize_t: bool = False
    mode_frequency: int = None
    phi_frequency: int = None


@dataclass
class Gray_Scott_Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    r: int = 8
    hidden_dim: int = 64
    num_epochs: int =15_00
    batch_size: int = 1
    lr: float = 5e-4
    kl_phi_weight: float =1e-3
    cons_weight: float = 0.15
    recon_weight: float = 3.
    save_dir: str = "results/gray_scott/test_run"
    print_every: int = 500
    ode_steps: int = 5
    process_noise: float = 1e-5
    cov_eps: float = 1e-6
    seed: int = 42
    sample_ratio: float = 0.1
    sigma: float = 0.001
    data_len: int = 100
    eval_data_len: int = 200
    mode_frequency: int = 2
    phi_frequency: int = 2
    dt: float = 0.1  # dt for data generation 
    resolution: tuple = (100,100)  # Added resolution parameter
    train_mode: str = "teacher_forcing"  # "teacher_forcing" or "autoreg" or "evolve"
    normalize_t: bool = False   

@dataclass
class Navier_Stokes_Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    r: int = 8
    hidden_dim: int = 128
    num_epochs: int =15_00
    batch_size: int = 1
    lr: float = 5e-4
    kl_phi_weight: float =1e-3
    cons_weight: float = 0.15
    recon_weight: float = 3.
    save_dir: str = "results/vorticity/test_run"
    print_every: int = 500
    ode_steps: int = 5
    process_noise: float = 1e-5
    cov_eps: float = 1e-6
    seed: int = 42
    sample_ratio: float = 0.1
    sigma: float = 0.001
    mode_frequency: int = 2
    phi_frequency: int = 2
    dt: float = 0.1  # dt for data generation 
    train_mode: str = "teacher_forcing"  # "teacher_forcing" or "autoreg" or "evolve"
    normalize_t: bool = False    
    data_len: int = 50
    eval_data_len: int = 99
    data_path: str = "dataset/navier_stokes_flow_dataset/vorticity.nc"

@dataclass
class Navier_Stokes_Cylinder_Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    r: int = 8
    hidden_dim: int = 128
    num_epochs: int =15_00
    batch_size: int = 1
    lr: float = 5e-4
    kl_phi_weight: float =1e-3
    cons_weight: float = 0.15
    recon_weight: float = 3.
    save_dir: str = "results/cylinder/test_run"
    print_every: int = 500
    ode_steps: int = 5
    process_noise: float = 1e-5
    cov_eps: float = 1e-6
    seed: int = 42
    sample_ratio: float = 0.1
    sigma: float = 0.001
    mode_frequency: int = 2
    phi_frequency: int = 2
    dt: float = 0.1  # dt for data generation 
    train_mode: str = "teacher_forcing"  # "teacher_forcing" or "autoreg" or "evolve"
    normalize_t: bool = False
 
    data_len: int = 150 
    eval_data_len: int = 199
    data_path: str = "dataset/navier_stokes_flow_dataset/cylinder.nc"
