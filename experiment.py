"""
Bad-to-Better residual policy experiment.

Good/bad actions are produced by two frozen randomly-initialised networks (RND-style).
Five methods are compared on a hard quadrant-split train/test:

  Zero            : no correction (floor)
  Direct          : one-shot residual to the good action
  Band+1-Random   : residual to random action in next band (high target variance)
  Band+1-Fixed    : residual to deterministic band-midpoint target (low variance)
  Band+1-Fixed+GT : Band+1-Fixed + cosine quality feedback + band index as inputs

Train states: state[0]>0 AND state[1]>0  (one quadrant of the unit sphere)
Test  states: state[0]<0 AND state[1]<0  (opposite quadrant — disjoint)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# ---------------------------------------------------------------------------
# 1.  RND-style frozen oracle / bad networks
# ---------------------------------------------------------------------------


class GoodNet(nn.Module):
    """Frozen randomly-initialised oracle: state -> good action direction."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class BadNet(nn.Module):
    """Frozen randomly-initialised shallow network: state -> noisy bad action."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor, noise_scale: float = 0.5) -> torch.Tensor:
        out = self.net(x)
        if noise_scale > 0:
            out = out + noise_scale * torch.randn_like(out)
        return F.normalize(out, dim=-1)


# -- GoodNet architecture variants ------------------------------------------


class HighFreqGoodNet(nn.Module):
    """Random Fourier feature GoodNet — high-frequency state→action mapping.
    The sinusoidal projection makes the function oscillate rapidly, so
    extrapolation across the quadrant split is nearly impossible for Direct."""

    def __init__(self, state_dim: int, action_dim: int, num_features: int = 128, omega: float = 10.0):
        super().__init__()
        self.omega = omega
        self.proj = nn.Linear(state_dim, num_features)
        self.out = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.out(torch.sin(self.omega * self.proj(x))), dim=-1)


class CompositionalGoodNet(nn.Module):
    """Smooth base + high-frequency perturbation.
    The smooth component extrapolates across the split; the HF component does not.
    Direct must predict both → fails on the HF part.
    Band+1 corrects incrementally: low bands are dominated by the smooth direction,
    high bands have smaller steps that limit HF-error accumulation."""

    def __init__(self, state_dim: int, action_dim: int, omega: float = 10.0, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.omega = omega
        self.smooth = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, action_dim),
        )
        self.hf_proj = nn.Linear(state_dim, 128)
        self.hf_out = nn.Sequential(
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, action_dim),
        )
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        smooth = self.smooth(x)
        hf = self.hf_out(torch.sin(self.omega * self.hf_proj(x)))
        return F.normalize(smooth + self.alpha * hf, dim=-1)


class SparseGoodNet(nn.Module):
    """Deep ReLU network (6 layers).  Many ReLU activations create a piecewise-linear
    function whose linear regions differ between the training and test quadrants,
    making it difficult to extrapolate even though it is continuous."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64, depth: int = 6):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(state_dim, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        layers.append(nn.Linear(hidden, action_dim))
        self.net = nn.Sequential(*layers)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def make_good_net(
    mode: str, state_dim: int, action_dim: int,
    omega: float = 10.0, alpha: float = 0.5,
) -> nn.Module:
    """Factory for GoodNet architecture variants."""
    if mode == "smooth":
        return GoodNet(state_dim, action_dim)
    if mode == "highfreq":
        return HighFreqGoodNet(state_dim, action_dim, omega=omega)
    if mode == "compositional":
        return CompositionalGoodNet(state_dim, action_dim, omega=omega, alpha=alpha)
    if mode == "sparse":
        return SparseGoodNet(state_dim, action_dim)
    raise ValueError(f"Unknown gen-mode: {mode}")


# ---------------------------------------------------------------------------
# 2.  Adaptive band edges — fit on training states only
# ---------------------------------------------------------------------------


def fit_band_edges(
    good_net: GoodNet,
    bad_net: BadNet,
    train_states: np.ndarray,
    device: torch.device,
    num_bands: int = 5,
) -> np.ndarray:
    """Empirical cos(bad, good) distribution split into equal-mass quantile bands."""
    states_t = torch.from_numpy(train_states).float().to(device)
    with torch.no_grad():
        a_good = good_net(states_t).cpu().numpy()
        a_bad = bad_net(states_t, noise_scale=0.5).cpu().numpy()

    cos_sims = np.einsum("ij,ij->i", a_good, a_bad)
    edges = np.quantile(cos_sims, np.linspace(0.0, 1.0, num_bands + 1))
    edges = np.unique(edges)
    if len(edges) < num_bands + 1:
        edges = np.linspace(float(cos_sims.min()), float(cos_sims.max()), num_bands + 1)
    return edges


def get_band_index(cos_sim: float, band_edges: np.ndarray) -> int:
    c = float(np.clip(cos_sim, band_edges[0], band_edges[-1]))
    for i in range(len(band_edges) - 1):
        if band_edges[i] <= c <= band_edges[i + 1]:
            return i
    return len(band_edges) - 2


# ---------------------------------------------------------------------------
# 3.  Fixed-target helper: geometric interpolation in the (a_curr, a_good) plane
# ---------------------------------------------------------------------------


def fixed_band_target(
    a_curr: np.ndarray,
    a_good: np.ndarray,
    target_cos: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Construct a unit vector v such that cos(v, a_good) == target_cos,
    lying in the same great-circle plane as (a_curr, a_good).

    This gives a deterministic, low-variance supervised target compared to
    rejection-sampling a random action that happens to land in a band.
    """
    # Orthogonal component of a_curr relative to a_good
    parallel = np.dot(a_curr, a_good) * a_good
    perp = a_curr - parallel
    perp_norm = np.linalg.norm(perp)
    if perp_norm < 1e-8:
        # a_curr is (anti-)parallel — pick any orthogonal direction
        perp = rng.standard_normal(a_curr.shape)
        perp -= np.dot(perp, a_good) * a_good
        perp_norm = np.linalg.norm(perp)
        if perp_norm < 1e-8:
            return a_good.copy()
    perp_unit = perp / perp_norm
    sin_t = np.sqrt(max(0.0, 1.0 - target_cos ** 2))
    v = target_cos * a_good + sin_t * perp_unit
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# 4.  Spatial train / test state split (quadrant — harder than half-space)
# ---------------------------------------------------------------------------


def generate_states_split(
    n_train: int,
    n_test: int,
    state_dim: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quadrant split:
      Train: state[0] > 0  AND  state[1] > 0   (~25% of unit sphere)
      Test : state[0] < 0  AND  state[1] < 0   (opposite quadrant)
    These are disjoint and far apart in state space.
    """
    train_states: list[np.ndarray] = []
    test_states: list[np.ndarray] = []
    while len(train_states) < n_train or len(test_states) < n_test:
        batch = rng.standard_normal((max(n_train, n_test) * 8, state_dim))
        for s in batch:
            s = s / np.linalg.norm(s)
            if s[0] > 0 and s[1] > 0 and len(train_states) < n_train:
                train_states.append(s)
            elif s[0] < 0 and s[1] < 0 and len(test_states) < n_test:
                test_states.append(s)
    return np.array(train_states[:n_train]), np.array(test_states[:n_test])


# ---------------------------------------------------------------------------
# 5.  Dataset generation
# ---------------------------------------------------------------------------


def _sample_action_in_band(
    a_opt: np.ndarray,
    target_band: int,
    band_edges: np.ndarray,
    rng: np.random.Generator,
    max_tries: int = 2000,
) -> np.ndarray:
    """Rejection-sample a unit action in a specific band relative to a_opt."""
    for _ in range(max_tries):
        v = rng.standard_normal(a_opt.shape)
        v = v / np.linalg.norm(v)
        if get_band_index(float(np.dot(v, a_opt)), band_edges) == target_band:
            return v
    # Fallback: fixed target at band midpoint
    mid = (band_edges[target_band] + band_edges[target_band + 1]) / 2.0
    return fixed_band_target(a_opt, a_opt, mid, rng)


def generate_dataset(
    states: np.ndarray,
    Y_good: np.ndarray,
    bad_net: BadNet,
    band_edges: np.ndarray,
    history_len: int,
    device: torch.device,
    rng: np.random.Generator,
    mode: str = "band+1-random",
    include_feedback: bool = False,
    top_band_direct: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build (X, Y_residual) for a given mode.

    X layout: [state | history(H*action_dim) | a_curr | (cos_sim, band_idx)?]
    a_curr is always at [state_dim + H*action_dim : state_dim + H*action_dim + action_dim].

    History is sampled from bad_net (H independent noisy forward passes per state) so it
    reflects what a bad policy would realistically produce, not random band placement.

    Modes:
      "direct"        -- a_curr = bad_net(state); Y = a_good - a_bad  (single fixed start)
      "band+1-random" -- a_curr rejection-sampled in current band; Y = random next-band target
      "band+1-fixed"  -- a_curr rejection-sampled in current band; Y = geometric midpoint of next band

    top_band_direct: when True, samples current_band over ALL bands (0..num_bands-1).
    At the top band the residual target is a_good directly, teaching the policy to complete
    the final correction and eliminating the OOD issue that arises during rollout.
    """
    n = len(states)
    action_dim = states.shape[1]
    num_bands = len(band_edges) - 1

    # --- Batch bad-network forward passes ---
    states_t = torch.from_numpy(states).float().to(device)
    # Repeat each state history_len times so one batched forward pass gives H bad actions/state
    states_rep_t = torch.from_numpy(
        np.repeat(states, history_len, axis=0)
    ).float().to(device)

    with torch.no_grad():
        a_bad_all = bad_net(states_t, noise_scale=0.5).cpu().numpy()          # (n, d)
        a_bad_hist = bad_net(states_rep_t, noise_scale=0.5).cpu().numpy()     # (n*H, d)
    a_bad_hist = a_bad_hist.reshape(n, history_len, action_dim)

    X_list, Y_list = [], []

    for i in range(n):
        a_opt = Y_good[i]
        a_bad = a_bad_all[i]

        # History: H independent noisy bad-network outputs for this state
        history = [a_bad_hist[i, h] for h in range(history_len)]

        if mode == "direct":
            # Always start from the actual bad-network action (not a band proxy)
            a_curr = a_bad
            cos_curr = float(np.dot(a_curr, a_opt))
            current_band = get_band_index(cos_curr, band_edges)
            residual = a_opt - a_curr

        else:
            # Band methods: sample a_curr from the appropriate band
            if top_band_direct:
                current_band = int(rng.integers(0, num_bands))   # 0..num_bands-1
            else:
                current_band = min(
                    get_band_index(float(np.dot(a_bad, a_opt)), band_edges), num_bands - 2
                )
            a_curr = _sample_action_in_band(a_opt, current_band, band_edges, rng)
            cos_curr = float(np.dot(a_curr, a_opt))

            at_top = top_band_direct and current_band >= num_bands - 1

            if mode == "band+1-random":
                if at_top:
                    residual = a_opt - a_curr
                else:
                    a_target = _sample_action_in_band(
                        a_opt, current_band + 1, band_edges, rng
                    )
                    residual = a_target - a_curr

            elif mode == "band+1-fixed":
                if at_top:
                    residual = a_opt - a_curr
                else:
                    next_b = current_band + 1
                    target_cos = (band_edges[next_b] + band_edges[next_b + 1]) / 2.0
                    a_target = fixed_band_target(a_curr, a_opt, target_cos, rng)
                    residual = a_target - a_curr
            else:
                raise ValueError(f"Unknown mode: {mode}")

        x = np.concatenate([states[i]] + history + [a_curr])
        if include_feedback:
            band_idx_norm = current_band / max(num_bands - 1, 1)
            x = np.concatenate([x, [cos_curr, band_idx_norm]])

        X_list.append(x)
        Y_list.append(residual)

    return (
        torch.tensor(np.array(X_list), dtype=torch.float32),
        torch.tensor(np.array(Y_list), dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# 6.  Evaluation helpers
# ---------------------------------------------------------------------------


def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def eval_policy(
    policy: nn.Module | None,
    X: torch.Tensor,
    Y: torch.Tensor,
    Y_good: np.ndarray,
    band_edges: np.ndarray,
    action_dim: int,
    history_len: int,
    device: torch.device,
    rollout_steps: int = 5,
    include_rollout: bool = True,
    has_feedback: bool = False,
) -> dict:
    """
    Evaluate a residual policy (None = Zero baseline).
    a_curr is always at [state_dim + H*action_dim : state_dim + H*action_dim + action_dim]
    regardless of whether feedback dims are appended.
    """
    n = len(X)
    X_np = X.numpy()
    Y_np = Y.numpy()

    if policy is None:
        pred_np = np.zeros_like(Y_np)
        mse = float(np.mean(Y_np ** 2))
    else:
        policy.eval()
        with torch.no_grad():
            pred_t = policy(X.to(device))
            mse = float(F.mse_loss(pred_t, Y.to(device)).item())
            pred_np = pred_t.cpu().numpy()

    # a_curr is always at this fixed slice (feedback is appended after, not before)
    curr_start = action_dim + history_len * action_dim
    curr_end = curr_start + action_dim

    band_jumps = np.empty(n, dtype=np.float64)
    exactly_one = 0
    sim_old_sum = sim_new_sum = 0.0

    for i in range(n):
        a_opt = Y_good[i]
        a_curr = X_np[i, curr_start:curr_end]
        a_new = a_curr + pred_np[i]

        sim_old = cosine_np(a_curr, a_opt)
        sim_new = cosine_np(a_new, a_opt)
        b_old = get_band_index(sim_old, band_edges)
        b_new = get_band_index(sim_new, band_edges)
        jump = b_new - b_old
        band_jumps[i] = jump
        if jump == 1:
            exactly_one += 1
        sim_old_sum += sim_old
        sim_new_sum += sim_new

    out = {
        "mse": mse,
        "mean_band_jump": float(band_jumps.mean()),
        "frac_exactly_one_up": float(exactly_one / n),
        "frac_any_improvement": float(np.mean(band_jumps > 0)),
        "mean_cos_before": sim_old_sum / n,
        "mean_cos_after_1step": sim_new_sum / n,
    }

    if include_rollout:
        out.update(
            _rollout_eval(
                policy, X_np, Y_good, action_dim, history_len, device,
                rollout_steps, band_edges=band_edges, has_feedback=has_feedback,
            )
        )
    return out


def _rollout_eval(
    policy: nn.Module | None,
    X_np: np.ndarray,
    Y_good: np.ndarray,
    action_dim: int,
    history_len: int,
    device: torch.device,
    n_steps: int = 5,
    cos_near_optimal: float = 0.99,
    band_edges: np.ndarray | None = None,
    has_feedback: bool = False,
) -> dict:
    """Apply residual n_steps times (rolling history) and measure final cosine vs optimal."""
    n = len(X_np)
    state_dim = action_dim
    hist_flat = history_len * action_dim
    curr_start = state_dim + hist_flat
    curr_end = curr_start + action_dim
    num_bands = (len(band_edges) - 1) if band_edges is not None else 5
    sims = np.empty(n, dtype=np.float64)

    for i in range(n):
        state = X_np[i, :state_dim]
        a_opt = Y_good[i]
        h = X_np[i, state_dim : state_dim + hist_flat].reshape(history_len, action_dim).copy()
        a = X_np[i, curr_start:curr_end].copy()

        for _ in range(n_steps):
            if policy is None:
                break
            if has_feedback and band_edges is not None:
                cos_fb = cosine_np(a, a_opt)
                band_idx_norm = get_band_index(cos_fb, band_edges) / max(num_bands - 1, 1)
                x_vec = np.concatenate([state, h.reshape(-1), a, [cos_fb, band_idx_norm]])
            else:
                x_vec = np.concatenate([state, h.reshape(-1), a])

            with torch.no_grad():
                r = policy(
                    torch.from_numpy(x_vec).float().unsqueeze(0).to(device)
                ).cpu().numpy()[0]
            a_new = a + r
            h = np.concatenate([h[1:], a[np.newaxis, :]], axis=0)
            a = a_new

        sims[i] = cosine_np(a, a_opt)

    return {
        "rollout_mean_cos": float(sims.mean()),
        "rollout_frac_near_optimal": float(np.mean(sims >= cos_near_optimal)),
    }


# ---------------------------------------------------------------------------
# 7.  Residual policy network
# ---------------------------------------------------------------------------


class ResidualPolicy(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# 8.  Training loop
# ---------------------------------------------------------------------------


def train_policy(
    policy: ResidualPolicy,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    Y_good_train: np.ndarray,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    Y_good_test: np.ndarray,
    band_edges: np.ndarray,
    action_dim: int,
    history_len: int,
    device: torch.device,
    epochs: int = 400,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    label: str = "",
    has_feedback: bool = False,
) -> dict:
    optimizer = optim.Adam(policy.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        policy.train()
        perm = torch.randperm(X_train.size(0))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, X_train.size(0), batch_size):
            idx = perm[i : i + batch_size]
            bx = X_train[idx].to(device)
            by = Y_train[idx].to(device)
            optimizer.zero_grad()
            loss = criterion(policy(bx), by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        if (epoch + 1) % 50 == 0:
            batch_avg = epoch_loss / max(n_batches, 1)
            tr = eval_policy(
                policy, X_train, Y_train, Y_good_train,
                band_edges, action_dim, history_len, device,
                include_rollout=False, has_feedback=has_feedback,
            )
            te = eval_policy(
                policy, X_test, Y_test, Y_good_test,
                band_edges, action_dim, history_len, device,
                rollout_steps=8, include_rollout=True, has_feedback=has_feedback,
            )
            print(
                f"  [{label}] Ep {epoch + 1}/{epochs} | "
                f"loss={batch_avg:.5f} | "
                f"tr MSE={tr['mse']:.5f} jump={tr['mean_band_jump']:.3f} f+1={tr['frac_exactly_one_up']:.3f} | "
                f"te MSE={te['mse']:.5f} jump={te['mean_band_jump']:.3f} f+1={te['frac_exactly_one_up']:.3f} | "
                f"roll8={te['rollout_mean_cos']:.3f} f>0.99={te['rollout_frac_near_optimal']:.3f}"
            )

    return eval_policy(
        policy, X_test, Y_test, Y_good_test,
        band_edges, action_dim, history_len, device,
        rollout_steps=8, include_rollout=True, has_feedback=has_feedback,
    )


# ---------------------------------------------------------------------------
# 9.  Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Bad-to-Better residual policy experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gen-mode", choices=["smooth", "highfreq", "compositional", "sparse"],
        default="smooth",
        help="GoodNet architecture. 'smooth'=Tanh MLP (Direct-friendly), "
             "'highfreq'=sinusoidal RFF (breaks all extrapolation), "
             "'compositional'=smooth+HF blend (smooth direction extrapolates, HF magnitude doesn't), "
             "'sparse'=deep 6-layer ReLU (piecewise-linear with quadrant-dependent activation patterns)",
    )
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--history-len", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--n-train", type=int, default=8_000)
    parser.add_argument("--n-test", type=int, default=2_000)
    parser.add_argument("--omega", type=float, default=10.0,
                        help="Sinusoidal frequency for highfreq / compositional modes")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="HF blend weight for compositional mode (0=smooth, 1=fully HF)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ACTION_DIM = args.action_dim
    HISTORY_LEN = args.history_len
    EPOCHS = args.epochs
    N_TRAIN = args.n_train
    N_TEST = args.n_test
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    INPUT_DIM = ACTION_DIM * (1 + HISTORY_LEN + 1)
    INPUT_DIM_FB = INPUT_DIM + 2

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print(
        f"Config: gen_mode={args.gen_mode}  action_dim={ACTION_DIM}  "
        f"history_len={HISTORY_LEN}  epochs={EPOCHS}  "
        f"omega={args.omega}  alpha={args.alpha}  seed={args.seed}"
    )

    # -- Frozen networks ------------------------------------------------------
    print(f"\nInitialising frozen GoodNet ({args.gen_mode}) / BadNet (shallow)...")
    good_net = make_good_net(
        args.gen_mode, ACTION_DIM, ACTION_DIM, omega=args.omega, alpha=args.alpha
    ).to(DEVICE)
    bad_net = BadNet(ACTION_DIM, ACTION_DIM).to(DEVICE)
    good_net.eval()
    bad_net.eval()

    # -- Quadrant state split -------------------------------------------------
    print("Generating quadrant-split states...")
    print("  Train: state[0]>0 AND state[1]>0  |  Test: state[0]<0 AND state[1]<0")
    train_states, test_states = generate_states_split(N_TRAIN, N_TEST, ACTION_DIM, rng)
    print(f"  Train: {len(train_states)} states | Test: {len(test_states)} states")

    tm = train_states.mean(axis=0)
    sm = test_states.mean(axis=0)
    split_cos = cosine_np(tm, sm)
    print(f"  Split hardness cos(mean_train, mean_test) = {split_cos:.4f}  (−1 = maximally different)")

    # -- Adaptive band edges --------------------------------------------------
    print("Fitting adaptive band edges from training states...")
    band_edges = fit_band_edges(good_net, bad_net, train_states, DEVICE)
    print(f"  Band edges: {np.array2string(band_edges, precision=4)}")

    # -- Ground-truth good actions (shared across methods) --------------------
    print("Pre-computing GoodNet actions for train / test states...")
    with torch.no_grad():
        Yg_train = good_net(torch.from_numpy(train_states).float().to(DEVICE)).cpu().numpy()
        Yg_test = good_net(torch.from_numpy(test_states).float().to(DEVICE)).cpu().numpy()

    # Diagnostic: how correlated are good actions across the split?
    mean_cos_train = np.einsum("ij,ij->i", Yg_train, Yg_train[::-1]).mean()
    cross_sample = np.einsum("ij,ij->i",
                             Yg_train[:min(N_TRAIN, N_TEST)],
                             Yg_test[:min(N_TRAIN, N_TEST)]).mean()
    print(f"  Mean cos(a_good_train_i, a_good_test_i) ≈ {cross_sample:.4f}  "
          f"(low = hard to extrapolate)")

    # -- Datasets -------------------------------------------------------------
    print("Generating datasets for all methods...")

    X_tr_dir, Y_tr_dir = generate_dataset(
        train_states, Yg_train, bad_net, band_edges, HISTORY_LEN, DEVICE, rng, mode="direct"
    )
    X_te_dir, Y_te_dir = generate_dataset(
        test_states, Yg_test, bad_net, band_edges, HISTORY_LEN, DEVICE, rng, mode="direct"
    )

    X_tr_b1rf, Y_tr_b1rf = generate_dataset(
        train_states, Yg_train, bad_net, band_edges, HISTORY_LEN, DEVICE, rng,
        mode="band+1-random", top_band_direct=True,
    )
    X_te_b1rf, Y_te_b1rf = generate_dataset(
        test_states, Yg_test, bad_net, band_edges, HISTORY_LEN, DEVICE, rng,
        mode="band+1-random", top_band_direct=True,
    )

    X_tr_b1ff, Y_tr_b1ff = generate_dataset(
        train_states, Yg_train, bad_net, band_edges, HISTORY_LEN, DEVICE, rng,
        mode="band+1-fixed", top_band_direct=True,
    )
    X_te_b1ff, Y_te_b1ff = generate_dataset(
        test_states, Yg_test, bad_net, band_edges, HISTORY_LEN, DEVICE, rng,
        mode="band+1-fixed", top_band_direct=True,
    )

    X_tr_full, Y_tr_full = generate_dataset(
        train_states, Yg_train, bad_net, band_edges, HISTORY_LEN, DEVICE, rng,
        mode="band+1-fixed", include_feedback=True, top_band_direct=True,
    )
    X_te_full, Y_te_full = generate_dataset(
        test_states, Yg_test, bad_net, band_edges, HISTORY_LEN, DEVICE, rng,
        mode="band+1-fixed", include_feedback=True, top_band_direct=True,
    )

    results: dict[str, dict] = {}

    # -- Zero -----------------------------------------------------------------
    print("\n--- Zero (no correction) ---")
    results["Zero"] = eval_policy(
        None, X_te_dir, Y_te_dir, Yg_test,
        band_edges, ACTION_DIM, HISTORY_LEN, DEVICE, rollout_steps=8, include_rollout=True,
    )
    z = results["Zero"]
    print(f"  cos_before={z['mean_cos_before']:.4f}  rollout8={z['rollout_mean_cos']:.4f}")

    # -- Direct ---------------------------------------------------------------
    print("\n--- Direct (bad → good, single shot) ---")
    p_dir = ResidualPolicy(INPUT_DIM, ACTION_DIM).to(DEVICE)
    results["Direct"] = train_policy(
        p_dir, X_tr_dir, Y_tr_dir, Yg_train, X_te_dir, Y_te_dir, Yg_test,
        band_edges, ACTION_DIM, HISTORY_LEN, DEVICE, EPOCHS, label="Direct",
    )

    # -- Band+1-Random+Full ---------------------------------------------------
    print("\n--- Band+1-Random+Full ---")
    p_b1rf = ResidualPolicy(INPUT_DIM, ACTION_DIM).to(DEVICE)
    results["B1-Rand+Full"] = train_policy(
        p_b1rf, X_tr_b1rf, Y_tr_b1rf, Yg_train, X_te_b1rf, Y_te_b1rf, Yg_test,
        band_edges, ACTION_DIM, HISTORY_LEN, DEVICE, EPOCHS, label="B1-Rand+F",
    )

    # -- Band+1-Fixed+Full ----------------------------------------------------
    print("\n--- Band+1-Fixed+Full ---")
    p_b1ff = ResidualPolicy(INPUT_DIM, ACTION_DIM).to(DEVICE)
    results["B1-Fixed+Full"] = train_policy(
        p_b1ff, X_tr_b1ff, Y_tr_b1ff, Yg_train, X_te_b1ff, Y_te_b1ff, Yg_test,
        band_edges, ACTION_DIM, HISTORY_LEN, DEVICE, EPOCHS, label="B1-Fixed+F",
    )

    # -- Band+1-Fixed+GT+Full -------------------------------------------------
    print("\n--- Band+1-Fixed+GT+Full ---")
    p_full = ResidualPolicy(INPUT_DIM_FB, ACTION_DIM).to(DEVICE)
    results["B1-GT+Full"] = train_policy(
        p_full, X_tr_full, Y_tr_full, Yg_train, X_te_full, Y_te_full, Yg_test,
        band_edges, ACTION_DIM, HISTORY_LEN, DEVICE, EPOCHS,
        label="B1-GT+Full", has_feedback=True,
    )

    # -- Comparison table -----------------------------------------------------
    W = 115
    print("\n" + "=" * W)
    print(f"FINAL COMPARISON  [gen_mode={args.gen_mode}  omega={args.omega}  alpha={args.alpha}]"
          f"  (test: state[0]<0 & state[1]<0)")
    print("=" * W)
    print(
        f"{'Method':<22} | {'MSE':>8} | {'MeanJump':>9} | {'Frac+1':>7} | "
        f"{'FracImpr':>9} | {'1-step Cos':>11} | {'8x Roll Cos':>12} | {'Frac>=0.99':>10}"
    )
    print("-" * W)
    for name, m in results.items():
        print(
            f"{name:<22} | {m['mse']:>8.5f} | {m['mean_band_jump']:>9.3f} | "
            f"{m['frac_exactly_one_up']:>7.3f} | {m['frac_any_improvement']:>9.3f} | "
            f"{m['mean_cos_after_1step']:>11.4f} | {m['rollout_mean_cos']:>12.4f} | "
            f"{m['rollout_frac_near_optimal']:>10.3f}"
        )
    print("=" * W)
