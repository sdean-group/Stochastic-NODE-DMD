import math
import torch
import torch.nn as nn
from torch.nn import functional as F
def gaussian_nll_loss(y, mu, logvar):
    return 0.5 * (logvar + (y - mu).pow(2) / logvar.exp() + math.log(2 * math.pi)).mean()
def relaxed_gaussian_nll(y, mu, logvar, alpha=0.3):
    return 0.5 * ((y - mu)**2 / logvar.exp() + alpha * logvar).mean()
_nll = nn.GaussianNLLLoss(eps=1e-6, reduction='mean')

def stochastic_loss_fn(mu_u, logvar_u, y_next, mu_phi, logvar_phi, mu_phi_hat, logvar_phi_hat, lambda_param, W, *,
              recon_weight: float, kl_phi_weight: float, cons_weight: float):
    var_u = torch.exp(logvar_u)
    if y_next.dim() == 4:  # (B, R, m, 2)
        nR = y_next.size(1)
        recon = sum(_nll(mu_u, y_next[:,r], var_u) for r in range(nR)) / nR
    else:
        recon = _nll(mu_u, y_next, var_u)
    
    kl_phi = -0.5 * torch.sum(1 + logvar_phi - mu_phi.pow(2) - logvar_phi.exp())
    cons_loss = consistency_loss(mu_phi_hat, logvar_phi_hat, mu_phi, logvar_phi, weight=cons_weight)
    loss = recon_weight * recon + kl_phi_weight * kl_phi  + cons_loss
    if not torch.isfinite(loss):
        raise ValueError(f"Non-finite loss detected: recon={recon}, kl={kl_phi}, cons={cons_loss}\nlambda={lambda_param}")
    return loss, {"recon": recon.item(), "kl": (kl_phi_weight * kl_phi).item(), "cons": cons_loss.item()}


def consistency_loss(obs_mu_phi, obs_logvar_phi, pred_mu_phi, pred_logvar_phi, weight=1.0, kl_scale=0.001):
    # Mean consistency (MSE)
    mse = torch.mean((obs_mu_phi - pred_mu_phi)**2)
    
    # print(f"obs_mu_phi magnitude: {torch.norm(obs_mu_phi, dim=-1).mean()}, pred_mu_phi magnitude: {torch.norm(pred_mu_phi, dim=-1).mean()}, mse: {mse.item()}")
    # Variance consistency (KL divergence: observed ~ N(obs_mu, obs_var) || predicted ~ N(pred_mu, pred_var))
    obs_var = torch.exp(obs_logvar_phi)  # (r, 2) or (B, r, 2)
    pred_var = torch.exp(pred_logvar_phi)  # (r, 2) or (B, r, 2)
    
    # KL divergence
    kl = 0.5 * torch.mean(
        torch.log(pred_var / obs_var) + (obs_var + (obs_mu_phi - pred_mu_phi)**2) / pred_var - 1
    )
    if not torch.isfinite(weight * (mse + kl_scale * kl)):
        raise ValueError(f"Non-finite loss detected: obs_mu_phi={obs_mu_phi}, obs_logvar_phi={obs_logvar_phi}, pred_mu_phi={pred_mu_phi}, pred_logvar_phi={pred_logvar_phi}")
    
    return weight * (mse + kl_scale * kl)

def loss_fn(u_pred, y_next, mu, logvar, lambda_param, l1_weight: float):
    # u_pred, mu, logvar, lambda_param = model(coords, y_prev, t_prev, t_next)
    recon_loss = F.mse_loss(u_pred, y_next)  # Predict next step
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    l1_loss = l1_weight * torch.mean(torch.abs(lambda_param))
    return recon_loss + 0.001 * kl_loss + l1_loss

