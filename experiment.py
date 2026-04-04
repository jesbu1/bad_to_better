import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- 1. Adaptive bands from empirical bad↔optimal cosine similarity ---


def random_unit_vector(shape, rng: np.random.Generator | None = None):
    if rng is None:
        v = np.random.randn(*shape)
    else:
        v = rng.standard_normal(shape)
    return v / np.linalg.norm(v)


def fit_band_edges_from_cosine_samples(
    num_samples: int, action_dim: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Segment the empirical distribution of cos(a_bad, a_opt) into 5 quantile bands
    (equal mass). Fit from random unit vectors only — no held-out test trajectories.
    """
    shape = (action_dim,)
    cos_sims = np.empty(num_samples, dtype=np.float64)
    for i in range(num_samples):
        a_opt = random_unit_vector(shape, rng)
        a_bad = random_unit_vector(shape, rng)
        cos_sims[i] = float(np.dot(a_bad, a_opt))
    edges = np.quantile(cos_sims, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # Ensure strictly increasing edges for non-degenerate bands
    edges = np.unique(edges)
    if len(edges) < 6:
        lo, hi = float(cos_sims.min()), float(cos_sims.max())
        edges = np.linspace(lo, hi, 6)
    return edges


def get_band_index(cos_sim: float, band_edges: np.ndarray) -> int:
    """Map cosine similarity to band index 0..4 using adaptive edges (length 6)."""
    c = float(np.clip(cos_sim, band_edges[0], band_edges[-1]))
    for i in range(5):
        if band_edges[i] <= c <= band_edges[i + 1]:
            return i
    # Fallback for numerical edge cases
    if c <= band_edges[0]:
        return 0
    return 4


def sample_action_in_band(
    a_opt, target_band_idx: int, band_edges: np.ndarray, rng: np.random.Generator
):
    """Rejection sample a unit action whose cos with a_opt falls in target_band_idx."""
    while True:
        a_rand = random_unit_vector(a_opt.shape, rng)
        cos_sim = float(np.dot(a_rand, a_opt))
        if get_band_index(cos_sim, band_edges) == target_band_idx:
            return a_rand


# --- 2. Dataset generation ---


def generate_trajectory_dataset(
    num_samples,
    band_edges: np.ndarray,
    history_len=3,
    action_dim=5,
    rng: np.random.Generator | None = None,
):
    """
    Supervised residual: current action in band b, target in band b+1 (same a_opt).
    """
    if rng is None:
        rng = np.random.default_rng()
    X = []
    Y_residual = []

    for _ in range(num_samples):
        state = rng.standard_normal(action_dim)
        a_opt = state / np.linalg.norm(state)

        history = []
        for _ in range(history_len):
            random_band = int(rng.integers(0, 4))
            history.append(sample_action_in_band(a_opt, random_band, band_edges, rng))

        current_band = int(rng.integers(0, 4))
        a_curr = sample_action_in_band(a_opt, current_band, band_edges, rng)

        target_band = current_band + 1
        a_target = sample_action_in_band(a_opt, target_band, band_edges, rng)

        residual = a_target - a_curr
        x = np.concatenate([state] + history + [a_curr])
        X.append(x)
        Y_residual.append(residual)

    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(
        np.array(Y_residual), dtype=torch.float32
    )


def mean_cosine_after_rollout(
    policy: nn.Module,
    X: torch.Tensor,
    action_dim: int,
    history_len: int,
    device: torch.device,
    n_steps: int = 5,
    cos_near_optimal: float = 0.99,
):
    """
    Apply the residual policy n_steps times: a <- a + r, rolling history forward each step.
    Returns mean cosine(sim(a, a_opt)) after the last step and fraction >= cos_near_optimal.
    """
    policy.eval()
    X_np = X.numpy()
    n = len(X_np)
    state_dim = action_dim
    hist_flat_dim = history_len * action_dim
    sims = np.empty(n, dtype=np.float64)
    for i in range(n):
        state = X_np[i, :state_dim]
        a_opt = state / np.linalg.norm(state)
        h = X_np[i, state_dim : state_dim + hist_flat_dim].reshape(history_len, action_dim).copy()
        a = X_np[i, -action_dim:].copy()
        for _ in range(n_steps):
            x = np.concatenate([state, h.reshape(-1), a])
            x_t = torch.from_numpy(x).float().unsqueeze(0).to(device)
            with torch.no_grad():
                pred = policy(x_t).cpu().numpy()[0]
            a_new = a + pred
            h = np.concatenate([h[1:], a[np.newaxis, :]], axis=0)
            a = a_new
        sims[i] = float(np.dot(a, a_opt) / (np.linalg.norm(a) * np.linalg.norm(a_opt)))
    return {
        "mean_cos_after_rollout": float(sims.mean()),
        "frac_near_optimal": float(np.mean(sims >= cos_near_optimal)),
    }


def eval_with_labels(
    policy: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    band_edges: np.ndarray,
    action_dim: int,
    history_len: int,
    device: torch.device,
    rollout_steps: int = 5,
    include_rollout: bool = True,
):
    policy.eval()
    with torch.no_grad():
        pred = policy(X.to(device))
        mse = float(nn.functional.mse_loss(pred, Y.to(device)).item())
        pred_np = pred.cpu().numpy()
    X_np = X.numpy()
    n = len(X_np)
    band_jumps = []
    exactly_one = 0
    sim_old_sum = 0.0
    sim_new_sum = 0.0

    for i in range(n):
        state = X_np[i][:action_dim]
        a_opt = state / np.linalg.norm(state)
        a_curr = X_np[i][-action_dim:]
        a_new = a_curr + pred_np[i]

        sim_old = float(np.dot(a_curr, a_opt) / (np.linalg.norm(a_curr) * np.linalg.norm(a_opt)))
        sim_new = float(np.dot(a_new, a_opt) / (np.linalg.norm(a_new) * np.linalg.norm(a_opt)))

        b_old = get_band_index(sim_old, band_edges)
        b_new = get_band_index(sim_new, band_edges)
        jump = b_new - b_old
        band_jumps.append(jump)
        if jump == 1:
            exactly_one += 1
        sim_old_sum += sim_old
        sim_new_sum += sim_new

    jumps = np.array(band_jumps, dtype=np.float64)
    out = {
        "mse": mse,
        "mean_band_jump": float(jumps.mean()),
        "frac_exactly_one_band_up": float(exactly_one / n),
        "mean_cos_old": sim_old_sum / n,
        "mean_cos_new": sim_new_sum / n,
        "frac_any_improvement": float(np.mean(jumps > 0)),
    }
    if include_rollout:
        out.update(
            mean_cosine_after_rollout(
                policy, X, action_dim, history_len, device, n_steps=rollout_steps
            )
        )
    return out


# --- 3. Policy ---


class ResidualPolicy(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# --- 4. Main ---

if __name__ == "__main__":
    ACTION_DIM = 5
    HISTORY_LEN = 3
    INPUT_DIM = ACTION_DIM + (HISTORY_LEN * ACTION_DIM) + ACTION_DIM

    train_rng = np.random.default_rng(42)
    test_rng = np.random.default_rng(43)

    print("Fitting adaptive band edges from training-only cosine samples (bad vs optimal)...")
    band_edges = fit_band_edges_from_cosine_samples(50_000, ACTION_DIM, train_rng)
    print(f"Band edges (cosine quantiles): {np.array2string(band_edges, precision=4)}")

    n_train = 8000
    n_test = 2000
    print(f"Generating datasets (train={n_train}, test={n_test}, disjoint RNG streams)...")
    X_train, Y_train = generate_trajectory_dataset(
        n_train, band_edges, HISTORY_LEN, ACTION_DIM, train_rng
    )
    X_test, Y_test = generate_trajectory_dataset(
        n_test, band_edges, HISTORY_LEN, ACTION_DIM, test_rng
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = ResidualPolicy(INPUT_DIM, ACTION_DIM).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    epochs = 300
    batch_size = 256

    print("Training residual policy...")
    for epoch in range(epochs):
        policy.train()
        permutation = torch.randperm(X_train.size(0))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i : i + batch_size]
            batch_x = X_train[indices].to(device)
            batch_y = Y_train[indices].to(device)
            optimizer.zero_grad()
            predictions = policy(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 50 == 0:
            train_mse = epoch_loss / max(n_batches, 1)
            tr = eval_with_labels(
                policy,
                X_train,
                Y_train,
                band_edges,
                ACTION_DIM,
                HISTORY_LEN,
                device,
                include_rollout=False,
            )
            te = eval_with_labels(
                policy,
                X_test,
                Y_test,
                band_edges,
                ACTION_DIM,
                HISTORY_LEN,
                device,
            )
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train_loss(batch_avg)={train_mse:.6f} | "
                f"train_MSE={tr['mse']:.6f} | test_MSE={te['mse']:.6f} | "
                f"test_mean_band_jump={te['mean_band_jump']:.3f} (ideal 1.0) | "
                f"test_frac_exactly+1={te['frac_exactly_one_band_up']:.3f} | "
                f"test_mean_cos: {te['mean_cos_old']:.3f} -> {te['mean_cos_new']:.3f} | "
                f"test_5x_rollout_mean_cos={te['mean_cos_after_rollout']:.3f} (ideal 1.0) "
                f"frac>0.99={te['frac_near_optimal']:.3f}"
            )

    print("\nFinal test evaluation (adaptive bands, held-out test set):")
    te = eval_with_labels(
        policy, X_test, Y_test, band_edges, ACTION_DIM, HISTORY_LEN, device
    )
    print(f"  Test MSE: {te['mse']:.6f}")
    print(f"  Mean band jump: {te['mean_band_jump']:.4f} (target: 1.0)")
    print(f"  Fraction moved exactly one band up: {te['frac_exactly_one_band_up']:.1%}")
    print(f"  Fraction any improvement (band): {te['frac_any_improvement']:.1%}")
    print(f"  Mean cosine vs optimal: {te['mean_cos_old']:.4f} -> {te['mean_cos_new']:.4f}")
    print(
        f"  After 5 iterative residuals (rolled history): mean cos vs optimal "
        f"{te['mean_cos_after_rollout']:.4f} (ideal 1.0), "
        f"fraction cos >= 0.99: {te['frac_near_optimal']:.1%}"
    )
