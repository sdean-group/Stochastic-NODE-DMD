import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import complex_block_matrix, reparameterize
from utils.ode import ode_euler_uncertainty
import math
class PositionalEncoding(nn.Module):
    def __init__(self, num_frequencies=8, include_input=True):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.freq_bands = 2 ** torch.linspace(0, num_frequencies - 1, num_frequencies) * math.pi
    def forward(self, coords):
        """
        coords: (m,2) or (B,m,2)
        returns: (m, 2*2*num_frequencies [+2 if include_input]) or batched version
        """
        if coords.dim() == 2:
            out = []
            if self.include_input:
                out.append(coords)
            for freq in self.freq_bands.to(coords.device):
                out.append(torch.sin(freq * coords))
                out.append(torch.cos(freq * coords))
            return torch.cat(out, dim=-1)
        elif coords.dim() == 3:
            B, m, _ = coords.shape
            out = []
            if self.include_input:
                out.append(coords)
            for freq in self.freq_bands.to(coords.device):
                out.append(torch.sin(freq * coords))
                out.append(torch.cos(freq * coords))
            return torch.cat(out, dim=-1)
        else:
            raise ValueError("coords must be (m,2) or (B,m,2)")

def stabilize_lambda(raw_lambda, min_decay=1e-3, w_scale=10.0):
    # raw_lambda: (...,2) -> (...,2)
    real_raw, imag_raw = raw_lambda[...,0], raw_lambda[...,1]
    real = -F.softplus(real_raw) - min_decay     
    imag = w_scale * torch.tanh(imag_raw)       
    return torch.stack([real, imag], dim=-1)

# ---------------------------
# Mode extractor
# ---------------------------
class ModeExtractor(nn.Module):
    def __init__(self, r: int, hidden_dim: int):
        super().__init__()
        self.r = r
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, r * 2),
        )

    def forward(self, coords):
        """
        coords: (m,2) or (B,m,2)
        returns: (m,r,2) or (B,m,r,2)
        """
        if coords.dim() == 2:
            m = coords.size(0)
            out = self.net(coords)               # (m, r*2)
            return out.view(m, self.r, 2)        # (m, r, 2)
        elif coords.dim() == 3:
            B, m, _ = coords.shape
            x = coords.reshape(B * m, 2)         # (B*m, 2)
            out = self.net(x).view(B, m, self.r, 2)
            return out                           # (B, m, r, 2)
        else:
            raise ValueError("coords must be (m,2) or (B,m,2)")
class ModeExtractor_PE(nn.Module):
    def __init__(self, r: int, hidden_dim: int, num_frequencies: int = 4):
        super().__init__()
        self.r = r
        self.posenc = PositionalEncoding(num_frequencies=num_frequencies)
        in_dim = (2 * (2 * num_frequencies) + 2)  # sin/cos pair * 2D + original coords if include_input=True
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, r * 2),
        )

    def forward(self, coords):
        coords_emb = self.posenc(coords)  # (m, emb_dim)
        if coords_emb.dim() == 2:
            m = coords_emb.size(0)
            out = self.net(coords_emb)
            return out.view(m, self.r, 2)
        elif coords_emb.dim() == 3:
            B, m, _ = coords_emb.shape
            x = coords_emb.reshape(B * m, -1)
            out = self.net(x).view(B, m, self.r, 2)
            return out
        else:
            raise ValueError("coords must be (m,2) or (B,m,2)")

# ---------------------------
# PhiEncoder
# ---------------------------
class PhiEncoder(nn.Module):
    def __init__(self, r: int, hidden_dim: int):
        super().__init__()
        self.r = r
        self.embed = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.pool = nn.Linear(hidden_dim, hidden_dim)
        self.phi_mu = nn.Linear(hidden_dim, r * 2)
        self.phi_logvar = nn.Linear(hidden_dim, r * 2)
        self.lambda_out = nn.Linear(hidden_dim, r * 2)

    def forward(self, coords, y):
        """
        coords, y: (m,2) / (B,m,2)
        returns:
          mu:           (r,2)   or (B,r,2)
          logvar:       (r,2)   or (B,r,2)
          lambda_param: (r,2)   or (B,r,2)
        """
        if coords.dim() != y.dim():
            raise ValueError("coords and y must have the same rank")
        if coords.dim() == 2:
            # (m,2)
            x = torch.cat([coords, y], dim=-1)       # (m,4)
            h = self.embed(x)                        # (m,H)
            pooled = h.mean(dim=0)                   # (H,)
            pooled = F.relu(self.pool(pooled))       # (H,)
            mu = self.phi_mu(pooled).view(self.r, 2)
            logvar = self.phi_logvar(pooled).view(self.r, 2)
            lambda_param = self.lambda_out(pooled).view(self.r, 2)
            lambda_param = stabilize_lambda(lambda_param)
            return mu, logvar, lambda_param
        elif coords.dim() == 3:
            # (B,m,2)
            B, m, _ = coords.shape
            x = torch.cat([coords, y], dim=-1)       # (B,m,4)
            h = self.embed(x.view(B * m, 4)).view(B, m, -1)  # (B,m,H)
            pooled = h.mean(dim=1)                   # (B,H)
            pooled = F.relu(self.pool(pooled))       # (B,H)
            mu = self.phi_mu(pooled).view(B, self.r, 2)
            logvar = self.phi_logvar(pooled).view(B, self.r, 2)
            lambda_param = self.lambda_out(pooled).view(B, self.r, 2)
            lambda_param = stabilize_lambda(lambda_param)
            return mu, logvar, lambda_param
        else:
            raise ValueError("coords/y must be (m,2) or (B,m,2)")

class PhiEncoder_PE(nn.Module):
    def __init__(self, r: int, hidden_dim: int, num_frequencies: int = 3):
        super().__init__()
        self.r = r
        self.posenc = PositionalEncoding(num_frequencies=num_frequencies)
        in_dim = (2 * (2 * num_frequencies) + 2) + 2  # encoded coords + y(2)
        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.pool = nn.Linear(hidden_dim, hidden_dim)
        self.phi_mu = nn.Linear(hidden_dim, r * 2)
        self.phi_logvar = nn.Linear(hidden_dim, r * 2)
        self.lambda_out = nn.Linear(hidden_dim, r * 2)
        self.pre_input = None
    def forward(self, coords, y):
        coords_emb = self.posenc(coords)
        if self.pre_input is not None:
            distance = torch.norm(y - self.pre_input, dim=-1)
            self.pre_input = y
        else:
            distance = torch.zeros(y.shape[0])
            self.pre_input = y
        if coords_emb.dim() == 2:
            x = torch.cat([coords_emb, y], dim=-1)
            h = self.embed(x)
            pooled = h.mean(dim=0)
            pooled = F.relu(self.pool(pooled))
            mu = self.phi_mu(pooled).view(self.r, 2)
            logvar = self.phi_logvar(pooled).view(self.r, 2)
            lambda_param = self.lambda_out(pooled).view(self.r, 2)
            lambda_param = stabilize_lambda(lambda_param)
            # print(f"input change: {distance.mean()}, latent : {h.flatten()}, pooled: {pooled.flatten()}")
            return mu, logvar, lambda_param
        elif coords_emb.dim() == 3:
            B, m, _ = coords_emb.shape
            x = torch.cat([coords_emb, y], dim=-1)
            h = self.embed(x.view(B * m, -1)).view(B, m, -1)
            pooled = h.mean(dim=1)
            pooled = F.relu(self.pool(pooled))
            mu = self.phi_mu(pooled).view(B, self.r, 2)
            logvar = self.phi_logvar(pooled).view(B, self.r, 2)
            lambda_param = self.lambda_out(pooled).view(B, self.r, 2)
            lambda_param = stabilize_lambda(lambda_param)
            return mu, logvar, lambda_param
        else:
            raise ValueError("coords/y must be (m,2) or (B,m,2)")
       
class ODENet(nn.Module):
    def __init__(self, r: int, hidden_dim: int):
        super().__init__()
        self.r = r
        self.net = nn.Sequential(
            nn.Linear(r * 2 + r * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, r * 2),
            # nn.Tanh(), # for small correction
        )
        # self.output_scale = 5.0  # <-- add this
    def forward(self, phi, lambda_param, t):
        """
        phi:           (r,2) or (B,r,2)
        lambda_param:  (r,2) or (B,r,2)   (broadcastable to phi)
        t:             scalar or (B,)
        returns:       (r,2) or (B,r,2)
        """
        lam_complex = lambda_param[..., 0] + 1j * lambda_param[..., 1]  # (B,r)
        phi_complex = phi[..., 0] + 1j * phi[..., 1]  # (B,r)
        drift_complex = lam_complex * phi_complex  # (B,r)
        drift = torch.stack([drift_complex.real, drift_complex.imag], dim=-1)  # (B,r,2)
        if phi.dim() == 2:
            # 비배치
            r, two = phi.shape
            assert r == self.r and two == 2
            lam = lambda_param
            if lam.dim() == 2:
                lam = lam
            elif lam.dim() == 1:
                lam = lam.view(self.r, 2)
            else:
                lam = lam.expand_as(phi)

            if not torch.is_tensor(t):
                t = torch.tensor(t, dtype=phi.dtype, device=phi.device)
            t_feat = t.view(1).to(phi.dtype).to(phi.device)     # (1,)
            x = torch.cat([phi.reshape(-1), lam.reshape(-1), t_feat], dim=0)  # (4r+1,)
            out = self.net(x)   
            correction = out.view(self.r, 2)
            # correction = out.view(self.r, 2) * self.output_scale
            dphi = drift + correction  # Residual structure
                                # (r*2,)
            return dphi

        elif phi.dim() == 3:
            # 배치
            B, r, two = phi.shape
            assert r == self.r and two == 2
            lam = lambda_param
            if lam.dim() == 2:
                lam = lam.unsqueeze(0).expand(B, -1, -1)        # (B,r,2)
            elif lam.dim() == 3:
                pass
            else:
                raise ValueError("lambda_param must be (r,2) or (B,r,2)")
            if not torch.is_tensor(t):
                t = torch.tensor(t, dtype=phi.dtype, device=phi.device)
            if t.dim() == 0:
                t = t.repeat(B)                                 # (B,)
            t_feat = t.view(B, 1).to(phi.dtype).to(phi.device)  # (B,1)

            x = torch.cat([phi.reshape(B, -1), lam.reshape(B, -1), t_feat], dim=1)  # (B, 4r+1)
            correction = self.net(x.float()).view(B, self.r, 2)
            dphi = drift + correction  # Residual structure
            # print(f"drift magnitude: {torch.norm(drift, dim=-1).mean()}, correction magnitude: {torch.norm(correction, dim=-1).mean()}")
            
            return dphi
        else:
            raise ValueError("phi must be (r,2) or (B,r,2)")

class Stochastic_NODE_DMD(nn.Module):
    def __init__(self, r: int, hidden_dim: int, ode_steps: int, process_noise: float, cov_eps: float, dt: float, mode_frequency=None, phi_frequency=None):
        super().__init__()
        self.r = r
        self.ode_func = ODENet(r, hidden_dim)
        # self.ode_func = ODE()
        if mode_frequency is None or type(mode_frequency) is not int:
            self.mode_net = ModeExtractor(r, hidden_dim)
        else:
            self.mode_net = ModeExtractor_PE(r, hidden_dim, num_frequencies=mode_frequency)
        if phi_frequency is None or type(phi_frequency) is not int:
            self.phi_net = PhiEncoder(r, hidden_dim)
        else:
            self.phi_net = PhiEncoder_PE(r, hidden_dim, num_frequencies=phi_frequency)
        self.process_noise = process_noise
        self.cov_eps = cov_eps
        self.ode_dt = dt / ode_steps

    def _complex_block_matrix_batched(self, W):
        """
        W: (B,m,r,2) or (m,r,2)
        returns M: (B, 2m, 2r) or (2m, 2r)
        """
        if W.dim() == 3:
            # 비배치
            return complex_block_matrix(W)   # (2m,2r)
        elif W.dim() == 4:
            B, m, r, _ = W.shape
            # complex_block_matrix가 배치 미지원이면 per-sample 처리
            M_list = [complex_block_matrix(W[i]) for i in range(B)]  # each: (2m,2r)
            return torch.stack(M_list, dim=0)  # (B, 2m, 2r)
        else:
            raise ValueError("W must be (m,r,2) or (B,m,r,2)")

    def forward(model, coords, y_prev, t_prev, t_next):
        mu_phi, logvar_phi, lambda_param = model.phi_net(coords, y_prev)
        
        mu_phi_next, cov_phi_next = ode_euler_uncertainty(
                model.ode_func, mu_phi, logvar_phi, lambda_param, t_prev, t_next,
                process_noise=model.process_noise, cov_eps=model.cov_eps, basic_dt=model.ode_dt,)
        W = model.mode_net(coords)
        M = model._complex_block_matrix_batched(W) if coords.dim()==3 else complex_block_matrix(W)
        # mean
        if coords.dim() == 3:
            B, m, _ = coords.shape
            mu_phi_flat = mu_phi_next.reshape(B, -1)
            mu_u_flat = torch.matmul(M, mu_phi_flat.unsqueeze(-1)).squeeze(-1)
            mu_u = mu_u_flat.view(B, m, 2)
        else:
            m = coords.size(0)
            mu_phi_flat = mu_phi_next.reshape(-1)
            mu_u_flat = M @ mu_phi_flat
            mu_u = mu_u_flat.view(m, 2)
        # cov
        cov_u = M @ cov_phi_next @ M.transpose(-1, -2)
        if coords.dim() == 3:
            var_u = torch.clamp(torch.diagonal(cov_u, dim1=-2, dim2=-1), min=model.cov_eps)
            B, m, _ = coords.shape
            logvar_u = torch.log(var_u.view(B, m, 2))
        else:
            var_u = torch.clamp(torch.diagonal(cov_u), min=model.cov_eps).view(m, 2)
            logvar_u = torch.log(var_u)
        # return mu_u, logvar_u, cov_u, mu_phi, logvar_phi, lambda_param, W #before 1029
        
        diag_var = torch.diagonal(cov_phi_next, dim1=-2, dim2=-1).clamp_min(1e-8)
        logvar_phi_next = torch.log(diag_var)
        logvar_phi_next = logvar_phi_next.view(logvar_phi.shape)
        return mu_u, logvar_u, cov_u, mu_phi_next, logvar_phi_next, lambda_param, W
    


