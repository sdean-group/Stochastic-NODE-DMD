import os
import sys
import argparse
import torch
import torch.optim as optim
from tqdm import trange
from models.node_dmd import Stochastic_NODE_DMD
from utils.losses import stochastic_loss_fn
from utils.utils import set_seed, ensure_dir, reparameterize_full
import random
import numpy as np
from torch.utils.data import DataLoader
from utils.plots import plot_loss


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
        raise ValueError(f"Unknown dataset type: {dataset_type}. Choose 'navier_stokes', 'gray_scott', or 'synthetic'")


def load_data_navier_stokes(cfg, load_synth, device):
    """Load data for Navier-Stokes dataset."""
    t_list, coords_list, y_list, *_ = load_synth(
        cfg.data_path, 
        sample_ratio=cfg.sample_ratio, 
        normalize_t=cfg.normalize_t, 
        device=device, 
        data_len=cfg.data_len
    )
    return t_list, coords_list, y_list


def load_data(cfg, load_synth, device):
    """Load data for Gray-Scott dataset."""
    t_list, coords_list, y_list, *_ = load_synth(
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
    return t_list, coords_list, y_list


def run_train(cfg, load_synth, SynthDataset, dataset_type: str):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    
    # Load data based on dataset type
    if dataset_type == "cylinder" or dataset_type == "vorticity":
        t_list, coords_list, y_list= load_data_navier_stokes(cfg, load_synth, device)
    else:
        t_list, coords_list, y_list = load_data(cfg, load_synth, device)
    
    dataset = SynthDataset(t_list, coords_list, y_list)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )   
    model = Stochastic_NODE_DMD(
        r=cfg.r,
        hidden_dim=cfg.hidden_dim,
        ode_steps=cfg.ode_steps,
        process_noise=cfg.process_noise,
        cov_eps=cfg.cov_eps,
        dt=cfg.dt,
        mode_frequency=cfg.mode_frequency,
        phi_frequency=cfg.phi_frequency
    ).to(device)

    # Save config to output directory
    ensure_dir(cfg.save_dir)
    config_path = os.path.join(cfg.save_dir, "Config.txt")

    with open(config_path, "w") as f:
        # Use vars() for snode_dmd (like train_SNode_DMD.py), dir() for others
        if dataset_type == "synthetic":
            for k, v in vars(cfg).items():
                f.write(f"{k}: {v}\n")
        else:
            for k in dir(cfg):
                if not k.startswith('_') and not callable(getattr(cfg, k)):
                    v = getattr(cfg, k)
                    f.write(f"{k}: {v}\n")

    initial_lr = cfg.lr
    final_lr = initial_lr * 0.2
    opt = optim.Adam(model.parameters(), lr=initial_lr)
    decay_rate = (final_lr / initial_lr) ** (1 / cfg.num_epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lr_lambda=lambda epoch: decay_rate ** epoch
    )
    best = float("inf")
    
    avg_loss_list = []
    for epoch in trange(cfg.num_epochs, desc="Training"):
        total_loss = 0.0
        num_batches = 0
        u_pred = None
        
        if cfg.train_mode == "teacher_forcing":
            teacher_prob = 1
        elif cfg.train_mode == "autoreg":
            teacher_prob = 0
        elif cfg.train_mode == "evolve":
            teacher_prob = min(1, 1 - (2*epoch / cfg.num_epochs))
        
        for batch in dataloader:
            t_prev, t_next, coords, y_next, y_prev = [x.to(device) for x in batch]
            # Initialize u_pred if None (for snode_dmd compatibility)
            if u_pred is None:
                u_pred = y_prev
            opt.zero_grad()
            
            if random.random() < teacher_prob:
                mu_u, logvar_u, cov_u, mu_phi, logvar_phi, lam, W = model(coords, y_prev, t_prev, t_next)
                u_pred = reparameterize_full(mu_u.detach(), cov_u.detach())
                with torch.no_grad():
                    mu_phi_hat, logvar_phi_hat, _ = model.phi_net(coords, y_next)
            else:
                if u_pred is None:
                    u_pred = y_prev
                mu_u, logvar_u, cov_u, mu_phi, logvar_phi, lam, W = model(coords, u_pred, t_prev, t_next)
                u_pred = reparameterize_full(mu_u.detach(), cov_u.detach())
                with torch.no_grad():
                    mu_phi_hat, logvar_phi_hat, _ = model.phi_net(coords, u_pred)
            
            loss, parts = stochastic_loss_fn(
                mu_u, logvar_u, y_next, mu_phi, logvar_phi, mu_phi_hat, logvar_phi_hat, lam, W,
                recon_weight=cfg.recon_weight,
                kl_phi_weight=cfg.kl_phi_weight,
                cons_weight=cfg.cons_weight * min((epoch / cfg.num_epochs), 1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            
            total_loss += loss.item()
            num_batches += 1

        avg = total_loss / num_batches if num_batches > 0 else float('inf')
        avg_loss_list.append(avg)
        scheduler.step()
        if epoch % cfg.print_every == 0:
            print(f"Epoch {epoch:04d} | avg_loss={avg:.6f} | lr={scheduler.get_last_lr()[0]:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'best_loss': best,
                'loss_list': avg_loss_list
            }, os.path.join(cfg.save_dir, f'model_{epoch}.pt'))
            plot_loss(avg_loss_list, cfg.save_dir)
        if avg < best:
            best = avg
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'best_loss': best,
                'loss_list': avg_loss_list
            }, os.path.join(cfg.save_dir, 'best_model.pt'))
            plot_loss(avg_loss_list, cfg.save_dir)
    
    torch.save({
        'epoch': cfg.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'best_loss': best,
        'loss_list': avg_loss_list
    }, os.path.join(cfg.save_dir, 'final_model.pt'))
    plot_loss(avg_loss_list, cfg.save_dir, "final_loss.png")

    print(f"Training complete. Final model saved at {os.path.join(cfg.save_dir, 'final_model.pt')}")
    print(f"Best model saved at {os.path.join(cfg.save_dir, 'best_model.pt')} with loss {best:.6f}")


def main():
    parser = argparse.ArgumentParser(description='Train Stochastic NODE-DMD model')
    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True,
        choices=['vorticity', 'cylinder', 'gray_scott', 'synthetic'],
        help='Dataset type: vorticity, cylinder, gray_scott, or synthetic'
    )
    args = parser.parse_args()
    
    # Get appropriate config and dataset functions
    Config, load_synth, SynthDataset = get_config_and_dataset(args.dataset)
    
    # Create config instance and run training
    cfg = Config()
    run_train(cfg, load_synth, SynthDataset, args.dataset)


if __name__ == "__main__":
    main()

