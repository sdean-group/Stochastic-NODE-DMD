import torch
from torch.autograd.functional import jacobian
import random
# PyTorch 2.x: functorch 통합 (torch.func)
try:
    from torch.func import jvp, vmap, jacfwd
    _HAS_TORCH_FUNC = True
except Exception:
    from torch.autograd.functional import jacobian as _jacobian
    _HAS_TORCH_FUNC = False

def ode_euler(func, phi0, lambda_param, t_start: float, t_end: float, steps: int):
    dt = (t_end - t_start) / steps
    phi = phi0
    t = t_start
    for _ in range(steps):
        dphi = func(phi, lambda_param, t)
        phi = phi + dt * dphi
        t += dt
    return phi


def ode_euler_uncertainty(
    func,
    mu0,
    logvar0,
    lambda_param,
    t_start,
    t_end,
    basic_dt: float,
    process_noise: float = 1e-5,
    cov_eps: float = 1e-6,
    # ↓ 추가 옵션 (기본값은 저비용 대각 전파)
    mode: str = "diag",            # "diag" | "full"
    backprop_mean: bool = True,
    backprop_cov: bool = False,    # 공분산 경로는 기본 no-grad로 가볍게
    max_step: int = 2000,
    update_cov: bool=True,
):
    """
    Drop-in 대체 버전:
      - mode="diag": 공분산의 대각만 전파 (빠름, 기본)
      - mode="full": A Σ A^T + Qd 이산화로 full-cov 전파 (정확, 다소 느림)
    반환값은 (mu_next, cov) 동일. diag 모드도 cov는 (2r,2r)의 대각 행렬로 반환.
    """
    # basic_dt = random.uniform(0.01, 0.5)
    if mu0.dim() == 3:
        B = mu0.shape[0]
        r = mu0.shape[1]
        assert mu0.shape[2] == 2
    elif mu0.dim() == 2:
        B = 1
        r = mu0.shape[0]
        assert mu0.shape[1] == 2
        mu0 = mu0.unsqueeze(0)          # (1,r,2)
        logvar0 = logvar0.unsqueeze(0)  # (1,r,2)
        lambda_param = lambda_param.unsqueeze(0) if (torch.is_tensor(lambda_param) and lambda_param.dim() == 1) else lambda_param
        t_start = torch.as_tensor(t_start, device=mu0.device, dtype=mu0.dtype).reshape(1)
        t_end   = torch.as_tensor(t_end,   device=mu0.device, dtype=mu0.dtype).reshape(1)
        # print(f"mu0 min: {mu0.min().item():.4f}, mu0 max: {mu0.max().item():.4f}, logvar0 min: {logvar0.min().item():.6f}, logvar0 max: {logvar0.max().item():.6f}")
    n = r * 2
    mu = mu0.reshape(B, n)
    var = logvar0.exp().reshape(B, n)
    device, dtype = mu.device, mu.dtype
    eye = torch.eye(n, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)

    t = t_start.clone()
    delta_t = t_end - t_start
    steps = torch.round(delta_t / basic_dt).int()
    steps = torch.clamp(steps, min=0)
    dt = delta_t / torch.clamp(steps.to(delta_t.dtype), min=1)
    max_steps = int(steps.max().item()) if B > 0 else 1

    def f_flat(x_flat):
        x = x_flat.reshape(B, r, 2)
        return func(x, lambda_param, t).reshape(B, n)

    for i in range(max_steps):
        mask = (torch.tensor(i, device=device) < steps).float()
        # ---- 평균 업데이트 (역전파 유지/차단 선택) ----
        if backprop_mean:
            phi_mu = mu.reshape(B, r, 2)
            dphi = func(phi_mu, lambda_param, t).reshape(B, n)
            # print(f"dphi min: {dphi.min().item():.4f}, dphi max: {dphi.max().item():.4f}")
            mu = mu + (dt[:, None] * dphi) * mask[:, None]
        else:
            with torch.no_grad():
                phi_mu = mu.reshape(B, r, 2)
                dphi = func(phi_mu, lambda_param, t).reshape(B, n)
                mu = mu + (dt[:, None] * dphi) * mask[:, None]

        if update_cov: 
            
            if mode == "diag":
                with torch.no_grad() if not backprop_cov else torch.enable_grad():
                    sqrt_var = torch.sqrt(torch.clamp(var, min=cov_eps))
                    V = torch.diag_embed(sqrt_var)  # [B, n, n]

                    if _HAS_TORCH_FUNC:
                        def _single(v):
                            _, Jv = jvp(f_flat, (mu,), (v,))
                            return v + dt[:, None] * Jv
                        AV = vmap(_single, in_dims=2, out_dims=2)(V)  # [B, n, n]
                        new_var = (AV * AV).sum(dim=2) + dt[:, None] * process_noise
                    else:
                        new_var = torch.zeros_like(var)
                        for b in range(B):
                            mu_b = mu[b]
                            t_b = t[b]
                            dt_b = dt[b]
                            lambda_b = lambda_param[b:b+1]
                            def f_flat_b(x_flat):
                                x = x_flat.reshape(r, 2)
                                return func(x.unsqueeze(0), lambda_b, t_b.unsqueeze(0)).squeeze(0).flatten()
                            J_b = _jacobian(f_flat_b, mu_b, create_graph=False)  # [n, n]
                            sqrt_var_b = sqrt_var[b]
                            V_b = torch.diag(sqrt_var_b)  # [n, n]
                            AV_b = V_b + dt_b * (J_b @ V_b)
                            new_var[b] = (AV_b * AV_b).sum(dim=1) + dt_b * process_noise

                    new_var = torch.clamp(new_var, min=cov_eps)
                    var = var * (1 - mask[:, None]) + new_var * mask[:, None]

            elif mode == "full":
                # Σ를 full로 유지: A = I + dt*J, Σ_next = A Σ A^T + dt*Q
                # 초기 Σ = diag(var)
                # (첫 스텝만 full Σ 구성, 이후부터 full 전파) 
                if var.dim() == 2:
                    cov = torch.diag_embed(var)
                else:
                    cov = var  # 이미 full-cov일 수 있음 (재호출 시)

                with torch.no_grad() if not backprop_cov else torch.enable_grad():
                    if _HAS_TORCH_FUNC:
                        J = vmap(jacfwd(f_flat))(mu)  # [B, n, n]
                        A = eye + dt[:, None, None] * J
                        new_cov = A @ cov @ A.transpose(1, 2) + dt[:, None, None] * process_noise * eye
                        new_cov = (new_cov + new_cov.transpose(1, 2)) / 2 + cov_eps * eye
                    else:
                        # fallback: loop over batch
                        new_cov = torch.zeros_like(cov)
                        for b in range(B):
                            mu_b = mu[b]
                            t_b = t[b]
                            dt_b = dt[b]
                            cov_b = cov[b]
                            lambda_b = lambda_param[b:b+1]
                            def f_flat_b(x_flat):
                                x = x_flat.reshape(r, 2)
                                return func(x.unsqueeze(0), lambda_b, t_b.unsqueeze(0)).squeeze(0).flatten()
                            J_b = _jacobian(f_flat_b, mu_b, create_graph=False)  # [n, n]
                            A_b = torch.eye(n, device=device, dtype=dtype) + dt_b * J_b
                            new_cov_b = A_b @ cov_b @ A_b.T + dt_b * process_noise * torch.eye(n, device=device, dtype=dtype)
                            new_cov_b = (new_cov_b + new_cov_b.T) / 2 + cov_eps * torch.eye(n, device=device, dtype=dtype)
                            new_cov[b] = new_cov_b

                cov = cov * (1 - mask[:, None, None]) + new_cov * mask[:, None, None]
                var = cov  # 다음 루프에서 그대로 사용

            else:
                raise ValueError("mode must be 'diag' or 'full'.")

        t += dt * mask

    mu_next = mu.reshape(mu0.shape)
    if mode == "diag":
        cov_out = torch.diag_embed(var)  # [B, n, n] 대각행렬 반환 (기존 타입과 동일)
    else:  # "full"
        cov_out = var  # full-cov 텐서
    
    if B == 1:
        mu_next = mu_next.squeeze(0)           # (r,2)
        cov_out = cov_out.squeeze(0)           # (n,n)
    return mu_next, cov_out

     