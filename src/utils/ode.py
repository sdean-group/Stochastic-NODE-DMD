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



def ode_euler_uncertainty_batch(
    func,
    mu0: torch.Tensor,        # (B,*S) or (*S)  — (*S)==(r,2) 등 비배치도 허용
    logvar0: torch.Tensor,    # (B,*S) or (*S)  — diagonal only
    lambda_param,             # (B,...) or (...)  (브로드캐스트 가능)
    t_start,                  # (B,) or scalar
    t_end,                    # (B,) or scalar
    *,
    process_noise: float = 1e-5,
    cov_eps: float = 1e-6,
    backprop_mean: bool = True,
    basic_dt: float = 0.1,
    max_step: int = 200,
    smallN_thresh: int = 64,          # N<=이면 jacrev+vmap 경로
    covariance_update_every: int = 1, #4, # k스텝마다 분산 업데이트(자코비안) 수행
    reuse_J_steps: int = 0,           # 계산한 J를 다음 s스텝 재사용(0이면 매번 새로계산)
    compute_diagJ: bool = True        # 1차항(2 dt var ⊙ diagJ) 포함 여부
):
    """
    Batched Euler + fast diagonal variance propagation.
    • 비배치 입력(*S)도 자동 배치(B=1)로 변환 후 처리, 마지막에 다시 squeeze.
    """
    device = mu0.device
    dtype  = mu0.dtype

    # --- 비배치 입력 자동 래핑 (B=1) ---
    unbatched = (mu0.dim() == 2)  # 보통 (*S)==(r,2)
    if unbatched:
        mu0      = mu0.unsqueeze(0)          # (1,*S)
        logvar0  = logvar0.unsqueeze(0)      # (1,*S)
        # lambda_param이 (r,2)... 처럼 배치축 없으면 붙여줌
        if torch.is_tensor(lambda_param) and (lambda_param.dim() >= 1) and (lambda_param.shape[0] != 1):
            lambda_param = lambda_param.unsqueeze(0)  # (1, ...)
        # t도 (1,)로
        t_start = torch.as_tensor(t_start, device=device, dtype=dtype).reshape(1)
        t_end   = torch.as_tensor(t_end,   device=device, dtype=dtype).reshape(1)
    B = mu0.shape[0]
    state_shape = mu0.shape[1:]
    N = int(torch.tensor(state_shape, device=device).prod().item()) if state_shape else 1

    # ---- states/vars (분산 경로는 그래프 분리) ----
    mu_flat = mu0.reshape(B, -1).contiguous()                               # (B,N)
    var     = torch.exp(logvar0.detach()).reshape(B, -1).contiguous().clone()  # (B,N)

    # ---- time grid ----
    def _to_B_vec(x):
        if torch.is_tensor(x):
            x = x.to(device=device, dtype=dtype)
            return x.reshape(-1) if x.dim() > 0 else x.reshape(1).expand(B)
        else:
            return torch.tensor([x], device=device, dtype=dtype).expand(B)

    t_start = _to_B_vec(t_start)   # (B,)
    t_end   = _to_B_vec(t_end)     # (B,)

    dT = (t_end - t_start)                                                  # (B,)
    dt_cap = torch.as_tensor(basic_dt, device=device, dtype=dtype)
    n  = torch.ceil(torch.clamp(torch.abs(dT) / dt_cap, min=1.0)).to(torch.long)
    n  = torch.clamp(n, max=max_step)                                       # (B,)
    dt = dT / n.clamp(min=1)                                                # (B,)
    t  = t_start.clone()                                                    # (B,)
    n_max = int(n.max().item())

    # ---- torch.func helpers ----
    has_func = hasattr(torch, "func")
    jacrev   = getattr(torch.func, "jacrev", None)
    vmap     = getattr(torch.func, "vmap",   None)

    # ---- batched drift ----
    def f_batch(mu_flat_in, lam, tt):
        # mu_flat_in: (B,N), lam: (B,...) or broadcastable, tt: (B,)
        phi = mu_flat_in.reshape((B,) + state_shape)             # (B,*S)
        return func(phi, lam, tt).reshape(B, -1)                 # (B,N)

    # ---- J 캐시 (재사용/업데이트 간격) ----
    J_prev         = None                    # (B,N,N)
    J_prev_valid_s = -1

    for s in range(n_max):
        alive = (s < n)                                                         # (B,)
        if not torch.any(alive): break
        dt_eff = (alive.to(dtype) * dt).unsqueeze(1)                            # (B,1)

        # ---- mean update ----
        if backprop_mean:
            dphi = f_batch(mu_flat, lambda_param, t)                            # (B,N)
            mu_flat = mu_flat + dt_eff * dphi
            t = t + dt_eff.squeeze(1)

        else:
            with torch.no_grad():
                dphi = f_batch(mu_flat, lambda_param, t)
                mu_flat = mu_flat + dt_eff * dphi
                t = t + dt_eff.squeeze(1)

        # ---- variance update scheduling ----
        need_cov_update = (s % max(1, covariance_update_every) == 0)
        can_reuse_J     = (J_prev is not None) and (s - J_prev_valid_s <= max(0, reuse_J_steps))

        # 항상 process_noise는 누적
        with torch.no_grad():
            var = torch.clamp(var + dt_eff * process_noise, min=cov_eps)

        if not need_cov_update and can_reuse_J:
            # 캐시된 J로 빠른 보정
            with torch.no_grad():
                idx = torch.nonzero(alive, as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    dt_a  = dt[idx].unsqueeze(1)                                # (Ba,1)
                    var_a = var[idx]                                           # (Ba,N)
                    J_a   = J_prev[idx]                                        # (Ba,N,N)
                    if compute_diagJ:
                        diagJ = torch.diagonal(J_a, dim1=-2, dim2=-1)          # (Ba,N)
                        var_a = var_a + 2.0 * dt_a * (var_a * diagJ)
                    JSJT_diag = torch.matmul(J_a.pow(2), var_a.unsqueeze(-1)).squeeze(-1)
                    var_new   = var_a + (dt_a ** 2) * JSJT_diag
                    var[idx]  = torch.clamp(var_new, min=cov_eps)
            continue

        # ---- 여기서만 J 계산 (업데이트 시점) ----
        with torch.no_grad():
            idx = torch.nonzero(alive, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue

        if (N <= smallN_thresh) and has_func and (jacrev is not None) and (vmap is not None):
            # vmap(jacrev)로 alive 한 번에
            mu_a  = mu_flat[idx]                                              # (Ba,N)
            var_a = var[idx]                                                  # (Ba,N)
            # lambda_param 브로드캐스트 정리
            if torch.is_tensor(lambda_param) and lambda_param.shape[:1] == (B,):
                lam_a = lambda_param[idx]
            else:
                lam_a = lambda_param
            t_a   = t[idx]

            def f_single(mu_i, lam_i, t_i):
                x = mu_i.reshape((1,) + state_shape)                           # (1,*S)
                out = func(x, lam_i, t_i.reshape(()))                          # (1,*S)
                return out.reshape(-1)                                         # (N,)

            J_a = torch.func.vmap(torch.func.jacrev(f_single))(mu_a, lam_a, t_a)  # (Ba,N,N)

            # 캐시 갱신
            if (J_prev is None) or (J_prev.shape != (B, N, N)):
                J_prev = torch.zeros((B, N, N), device=device, dtype=dtype)
            J_prev.index_copy_(0, idx, J_a)
            J_prev_valid_s = s

            with torch.no_grad():
                dt_a = dt[idx].unsqueeze(1)
                if compute_diagJ:
                    diagJ = torch.diagonal(J_a, dim1=-2, dim2=-1)             # (Ba,N)
                    var_a = var_a + 2.0 * dt_a * (var_a * diagJ)
                JSJT_diag = torch.matmul(J_a.pow(2), var_a.unsqueeze(-1)).squeeze(-1)
                var_new   = var_a + (dt_a ** 2) * JSJT_diag
                var[idx]  = torch.clamp(var_new, min=cov_eps)

        else:
            # per-sample 경로 (alive만 loop; vectorize=True)
            J_list = []
            for i in idx.tolist():
                mu_i  = mu_flat[i].detach()                                    # (N,)
                var_i = var[i]
                # lambda_param 단일/배치 모두 대응
                lam_i = lambda_param[i] if (torch.is_tensor(lambda_param) and lambda_param.shape[:1] == (B,)) else lambda_param
                t_i   = t[i]
                dt_i  = dt[i]

                def f_flat_i(x_flat):
                    x = x_flat.reshape((1,) + state_shape)
                    out = func(x, lam_i, t_i)                                   # (1,*S)
                    return out.reshape(-1)                                      # (N,)

                J_i = torch.autograd.functional.jacobian(
                    f_flat_i, mu_i, create_graph=False, vectorize=True
                )                                                                # (N,N)
                J_list.append(J_i.unsqueeze(0))

                with torch.no_grad():
                    var_i_new = var_i
                    if compute_diagJ:
                        diagJ = torch.diagonal(J_i)                              # (N,)
                        var_i_new = var_i_new + 2.0 * dt_i * (var_i_new * diagJ)
                    JSJT_diag = torch.matmul(J_i.pow(2), var_i_new)              # (N,)
                    var_i_new = var_i_new + (dt_i ** 2) * JSJT_diag
                    var[i]    = torch.clamp(var_i_new, min=cov_eps)

            if len(J_list) > 0:
                J_a = torch.cat(J_list, dim=0)                                   # (Ba,N,N)
                if (J_prev is None) or (J_prev.shape != (B, N, N)):
                    J_prev = torch.zeros((B, N, N), device=device, dtype=dtype)
                J_prev.index_copy_(0, idx, J_a)
                J_prev_valid_s = s

    # ---- 출력 복원 (비배치였다면 squeeze) ----
    mu_next = mu_flat.reshape((B,) + state_shape)
    cov_out = torch.diag_embed(var)                                            # (B,N,N)
    if unbatched:
        mu_next = mu_next.squeeze(0)           # (*S)
        cov_out = cov_out.squeeze(0)           # (N,N)
    return mu_next, cov_out


# def ode_euler_uncertainty(
#     func,
#     mu0,
#     logvar0,
#     lambda_param,
#     t_start,
#     t_end,
#     basic_dt: float,
#     process_noise: float = 1e-5,
#     cov_eps: float = 1e-6,
#     # ↓ 추가 옵션 (기본값은 저비용 대각 전파)
#     mode: str = "diag",            # "diag" | "full"
#     backprop_mean: bool = True,
#     backprop_cov: bool = False,    # 공분산 경로는 기본 no-grad로 가볍게
#     max_step: int = 2000,
#     update_cov: bool=True,
# ):
#     """
#     Drop-in 대체 버전:
#       - mode="diag": 공분산의 대각만 전파 (빠름, 기본)
#       - mode="full": A Σ A^T + Qd 이산화로 full-cov 전파 (정확, 다소 느림)
#     반환값은 (mu_next, cov) 동일. diag 모드도 cov는 (2r,2r)의 대각 행렬로 반환.
#     """
#     # basic_dt = random.uniform(0.01, 0.5)
#     # print(f"shape of mu0: {mu0.shape}, logvar0: {logvar0.shape}, lambda_param:{lambda_param.shape} t_start: {t_start.shape}, t_end: {t_end.shape}, basic_dt: {basic_dt}")
#     # print(f"func(phi_mu, lambda_param, t) output shape: {func(mu0, lambda_param, t_start).shape}")
#     steps = int(round((t_end - t_start) / basic_dt))
#     dt = (t_end - t_start) / max(1, steps)
#     mu_flat = mu0.flatten()
#     var = torch.exp(logvar0).flatten()  # (2r,)
#     device, dtype = mu_flat.device, mu_flat.dtype
#     n = mu_flat.numel()
#     eye = torch.eye(n, device=device, dtype=dtype)

#     t = t_start
#     # print(f"t start: {t_start:.4f} to t end: {t_end:.4f} in {steps} steps of dt={dt:.4f}, basic_dt={basic_dt:.4f}, |t_end-t_start|={abs(t_end - t_start)/basic_dt:.4f}")
#     # print("ODE step start t=", t_start, " to ", t_end)
#     for _ in range(steps):
#         # ---- 평균 업데이트 (역전파 유지/차단 선택) ----
#         # print(f"ODE step t={t:.4f} dt={dt:.4f}")
#         if backprop_mean:
#             phi_mu = mu_flat.reshape(mu0.shape)
#             dphi = func(phi_mu, lambda_param, t).flatten()
#             mu_flat = mu_flat + dt * dphi
            
#         else:
#             with torch.no_grad():
#                 phi_mu = mu_flat.reshape(mu0.shape)
#                 dphi = func(phi_mu, lambda_param, t).flatten()
#                 mu_flat = mu_flat + dt * dphi

#         # ---- 공분산(또는 분산) 업데이트 ----
#         # f_flat(x) = func(x_reshaped, lam, t).flatten()
#         def f_flat(x_flat):
#             x = x_flat.reshape(mu0.shape)
#             return func(x, lambda_param, t).flatten()
#         if update_cov:
#             # 매 스텝마다 공분산 갱신
            
#             if mode == "diag":
#                 # Σ = diag(var)만 유지. A = I + dt*J.
#                 # diag(A Σ A^T) = rowwise_sum( (A V) ∘ (A V) ),
#                 #   where V = diag(sqrt(var)) (각 열이 가중 단위벡터)
#                 with torch.no_grad() if not backprop_cov else torch.enable_grad():
#                     sqrt_var = torch.sqrt(torch.clamp(var, min=cov_eps))
#                     # V_cols: (n, n) 각 열이 sqrt(var) 단위기저
#                     # 메모리: n^2. n=2r이므로 r<=128 규모면 충분히 경량.
#                     V_cols = torch.diag(sqrt_var)  # (n, n)

#                     if _HAS_TORCH_FUNC:
#                         # vmap으로 각 열에 대해 jvp(f, mu; v) 병렬 평가
#                         cols = V_cols.t()  # (n, n) => 각 row가 한 column 벡터
#                         def _single(v_col):
#                             # jvp returns (f(mu), J v_col)
#                             _, Jv = jvp(f_flat, (mu_flat,), (v_col,))
#                             return v_col + dt * Jv  # A v_col
#                         AV_cols = vmap(_single)(cols)  # (n, n)
#                         AV = AV_cols.t()  # (n, n) 열 방면으로 정렬
#                     else:
#                         # torch.func 없으면 보수적으로 전체 J 계산 (fallback)
#                         J = _jacobian(f_flat, mu_flat, create_graph=False)  # (n, n)
#                         AV = V_cols + dt * (J @ V_cols)

#                     # diag(A Σ A^T) = sum_j AV[:, j]^2
#                     var = (AV * AV).sum(dim=1) + dt * process_noise
#                     var = torch.clamp(var, min=cov_eps)

#             elif mode == "full":
#                 # Σ를 full로 유지: A = I + dt*J, Σ_next = A Σ A^T + dt*Q
#                 # 초기 Σ = diag(var)
#                 # (첫 스텝만 full Σ 구성, 이후부터 full 전파)
#                 if not isinstance(var, torch.Tensor) or var.dim() == 1:
#                     cov = torch.diag(var).clone()
#                 else:
#                     cov = var  # 이미 full-cov일 수 있음 (재호출 시)

#                 if _HAS_TORCH_FUNC:
#                     with torch.no_grad() if not backprop_cov else torch.enable_grad():
#                         J = jacfwd(f_flat)(mu_flat)  # (n, n)
#                         A = eye + dt * J
#                         cov = A @ cov @ A.T + dt * process_noise * eye
#                         cov = (cov + cov.T) / 2 + cov_eps * eye
#                 else:
#                     # fallback: autograd.functional.jacobian
#                     with torch.no_grad() if not backprop_cov else torch.enable_grad():
#                         J = _jacobian(f_flat, mu_flat, create_graph=False)  # (n, n)
#                         A = eye + dt * J
#                         cov = A @ cov @ A.T + dt * process_noise * eye
#                         cov = (cov + cov.T) / 2 + cov_eps * eye

#                 var = cov  # 다음 루프에서 그대로 사용

#             else:
#                 raise ValueError("mode must be 'diag' or 'full'.")

#         t += dt

#     mu_next = mu_flat.reshape(mu0.shape)
#     if mode == "diag":
#         cov_out = torch.diag(var)  # (n, n) 대각행렬 반환 (기존 타입과 동일)
#     else:  # "full"
#         cov_out = var  # full-cov 텐서
#     print(f"mu_next.shape: {mu_next.shape}, cov_out.shape: {cov_out.shape}")
#     return mu_next, cov_out

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

     