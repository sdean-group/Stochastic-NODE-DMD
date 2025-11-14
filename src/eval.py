# ============= eval_unified.py ==================
import os
import argparse
import torch
import torch.nn.functional as F
from models.node_dmd import Stochastic_NODE_DMD
from utils.utils import ensure_dir, eval_uncertainty_metrics, find_subset_indices
from utils.utils import FeedMode, compute_vmin_vmax, summarize_and_dump, load_config_from_file
from utils.plots import plot_reconstruction
import imageio

def get_config_and_dataset(dataset_type: str):
    """Import the appropriate config and dataset based on dataset type."""
    if dataset_type == "vorticity":
        from config.config import Navier_Stokes_Config
        from dataset.navier_stokes_flow import load_synth, SynthDataset
        return Navier_Stokes_Config, load_synth, SynthDataset
    elif dataset_type == "cylinder":
        from config.config import Navier_Stokes_Cylinder_Config
        from dataset.navier_stokes_flow import load_synth, SynthDataset
        return Navier_Stokes_Cylinder_Config, load_synth, SynthDataset
    elif dataset_type == "gray_scott":
        from config.config import Gray_Scott_Config
        from dataset.gray_scott import load_synth, SynthDataset
        return Gray_Scott_Config, load_synth, SynthDataset
    elif dataset_type == "synthetic":
        from config.config import Synthetic_Dataset_Config
        from dataset.generate_synth_dataset import load_synth, SynthDataset
        return Synthetic_Dataset_Config, load_synth, SynthDataset
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Choose 'vorticity', 'cylinder', 'gray_scott', or 'synthetic'")


def load_data_navier_stokes(cfg, load_synth, device):
    """Load data for Navier-Stokes dataset."""
    t_list, coords_list, y_list, y_true_full_list, coords_full = load_synth(
        cfg.data_path,
        sample_ratio=cfg.sample_ratio,
        normalize_t=cfg.normalize_t,
        device=device
    )
    return t_list, coords_list, y_list, y_true_full_list, coords_full

def load_data(cfg, load_synth, device):
    """Load data for Gray-Scott dataset."""
    data = load_synth(
        device, 
        T=cfg.data_len, 
        norm_T=cfg.data_len, 
        resolution=cfg.resolution, 
        sample_ratio=cfg.sample_ratio, 
        sigma=cfg.sigma, 
        dt=cfg.dt, 
        seed=cfg.seed, 
        normalize_t=cfg.normalize_t
    )
    t_list = data[0]
    coords_list = data[1]
    y_list = data[2]
    y_true_full_list = data[3]
    coords_full = data[4]
    return t_list, coords_list, y_list, y_true_full_list, coords_full

def _prepare_model(cfg, model_name="best_model.pt") -> Stochastic_NODE_DMD:
    """Prepare model based on dataset type."""
    device = torch.device(cfg.device)
    
    # Model initialization - use keyword arguments for all datasets (they're all compatible)
    model = Stochastic_NODE_DMD(
        cfg.r, cfg.hidden_dim, cfg.ode_steps, cfg.process_noise, cfg.cov_eps, cfg.dt,
        mode_frequency=cfg.mode_frequency,
        phi_frequency=cfg.phi_frequency
    ).to(device)
    
    ckpt = torch.load(os.path.join(cfg.save_dir, model_name), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    loss = ckpt["best_loss"]
    print(f"best loss of {loss}")
    return model


def _prepare_data(cfg, load_synth, dataset_type: str):
    """Prepare data based on dataset type."""
    device = torch.device(cfg.device)
    if dataset_type == "vorticity" or dataset_type == "cylinder":
        return load_data_navier_stokes(cfg, load_synth, device)
    else:  # synthetic
        return load_data(cfg, load_synth, device)



@torch.no_grad()
def run_eval(cfg, dataset_type: str, mode: str = "teacher_forcing", model_name: str = "best_model.pt"):
    _, load_synth, _ = get_config_and_dataset(dataset_type)
    
    # Prepare data
    feed_mode = FeedMode(mode)
    data = _prepare_data(cfg, load_synth, dataset_type)
    
    # Unpack data based on dataset type
    # if dataset_type == "synthetic":
    #     t_list, coords_list, y_list, y_true_list, y_true_full_list, coords_full, *_ = data
    # else:  # navier_stokes or gray_scott
    t_list, coords_list, y_list, y_true_full_list, coords_full = data
    
    model = _prepare_model(cfg, model_name=model_name)
    vmin, vmax = compute_vmin_vmax(y_true_full_list)

    coords_idx = find_subset_indices(coords_full, coords_list[0])

    mse_full_all = []
    calib_all = []
    out_dir = os.path.join(cfg.save_dir, f"{feed_mode.value}_reconstruction")
    ensure_dir(out_dir)
    
    # Initialize y_in (AUTOREG only)
    y_pred_chain = y_true_full_list[0] if feed_mode == FeedMode.AUTOREG else None
    frames = []
    frame = plot_reconstruction(coords_full, 0, y_true_full_list[0], y_true_full_list[0], 0, out_dir, vmin, vmax)
    frames.append(frame)
    
    # Main loop
    for i in range(1, cfg.data_len):
        coords = coords_full
        y_true = y_true_full_list[i]
        t_prev = float(t_list[i - 1])
        t_next = float(t_list[i])

        if feed_mode == FeedMode.TEACHER:
            y_in = y_true_full_list[i - 1]  # ground-truth teacher forcing
        else:
            y_in = y_pred_chain

        mu_u, logvar_u, cov_u, mu_phi, logvar_phi, lam, W = model(coords, y_in, t_prev, t_next)
        
        # Update autoreg chain
        if feed_mode == FeedMode.AUTOREG:
            y_pred_chain = mu_u

        # MSE and plot
        mse = F.mse_loss(mu_u, y_true).item()
        mse_full_all.append(mse)
        frame = plot_reconstruction(coords, i, y_true, mu_u, mse, out_dir, vmin, vmax)
        frames.append(frame)
        
        # Uncertainty calibration on observed subset (with noise)
        y_obs = y_list[i]  # noisy measurement at t_next (subset coords_list[0])
        mu_sub = mu_u[coords_idx]
        logvar_sub = logvar_u[coords_idx]

        metrics = eval_uncertainty_metrics(y_obs, mu_sub, logvar_sub)

        metrics["time_index"] = i
        metrics["mse_full"] = mse
        calib_all.append(metrics)
        
    # Save final results
    imageio.mimsave(f"{out_dir}/reconstruction.gif", frames, fps=10)
    summarize_and_dump(calib_all, mse_full_all, out_dir, feed_mode)


def find_config_file(config_dir: str, Config) -> str:
    """Find config file in directory, trying multiple possible names."""
    # Possible config file names (in order of preference)
    possible_names = [
        "Config.txt",  # Standard name used by training scripts
        f"{Config.__name__}.txt",  # Class name based (e.g., "Gray_Scott_Config.txt")
    ]
    
    for name in possible_names:
        config_path = os.path.join(config_dir, name)
        if os.path.exists(config_path):
            return config_path
    
    # If none found, raise error with helpful message
    raise FileNotFoundError(
        f"Config file not found in {config_dir}. "
        f"Tried: {', '.join(possible_names)}. "
        f"Please ensure the config file exists in the specified directory."
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate SNode DMD with config from directory.")
    parser.add_argument("--config_dir", type=str, required=True, help="Directory containing Config.txt")
    parser.add_argument("--dataset", type=str, required=True, choices=['vorticity', 'cylinder', 'gray_scott', 'synthetic'],
                        help="Dataset type: vorticity, cylinder, gray_scott, or synthetic")
    parser.add_argument("--ckpt_name", type=str, default="best_model.pt", help="Checkpoint name")
    
    args = parser.parse_args()

    # Get config class
    Config, _, _ = get_config_and_dataset(args.dataset)
    
    # Find and load config file
    config_path = find_config_file(args.config_dir, Config)
    cfg = load_config_from_file(config_path, Config)
    
    # Override save_dir with config_dir (in case it changed)
    cfg.save_dir = args.config_dir

    run_eval(cfg, args.dataset, mode="teacher_forcing", model_name=args.ckpt_name)
    run_eval(cfg, args.dataset, mode="autoreg", model_name=args.ckpt_name)

if __name__ == "__main__":
    main()

