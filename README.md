# Bad-to-Better: Residual Policy Correction Experiment

A supervised learning experiment studying how a **residual policy** can incrementally improve a bad policy's actions toward a good policy's actions — without retraining the underlying policies — using structured cosine-similarity bands as a curriculum.

## Motivation

Given a trajectory of suboptimal ("bad") actions from a fixed bad policy, can we train a small corrector network that, when applied iteratively, reliably recovers near-optimal ("good") actions? The core hypothesis is that **incremental band-by-band correction** generalizes better out-of-distribution than a single-shot "predict the full correction" approach.

## How It Works

### Environment

There is no simulator. States and actions are vectors on the **unit sphere** in `ACTION_DIM = 8` dimensions. The "environment" is defined entirely by two frozen networks:

- **GoodNet** — a 2-layer Tanh MLP (hidden=64) that maps a state to its *oracle* good action direction.
- **BadNet** — a 1-layer ReLU MLP (hidden=24, with additive noise at inference) that maps the same state to a structurally worse, noisier action.

Both networks are **randomly initialized once and never trained** (all `requires_grad=False`). This mirrors Random Network Distillation (RND): a frozen random network provides a consistent, structured signal without any human-designed reward. The architecture asymmetry (deeper + Tanh vs shallower + ReLU + noise) ensures GoodNet systematically produces actions more aligned with any ground truth structure than BadNet.

```
GoodNet(state) → normalize(2-layer Tanh MLP) → a_good  (unit vector)
BadNet(state)  → normalize(1-layer ReLU MLP + noise)   → a_bad   (unit vector, noisy)
```

### Quality Metric: Cosine Similarity

The optimality of any action `a` for a given state `s` is measured as:

```
quality(a, s) = cos(a, GoodNet(s))
```

A quality of 1.0 means the action perfectly matches the oracle. A quality near 0 or negative means the action is uninformative or counter-productive.

### Adaptive Bands

Instead of pre-defining fixed cosine similarity thresholds, **5 bands are fit empirically** from the training data. For every training state, we compute `cos(BadNet(state), GoodNet(state))` and divide the resulting distribution into 5 equal-mass quantile intervals:

```
Band 0: [min_cos,  q20]   ← worst
Band 1: [q20,      q40]
Band 2: [q40,      q60]
Band 3: [q60,      q80]
Band 4: [q80,  max_cos]   ← best achievable by bad policy
```

Band edges are computed on training states only — test states never touch the band-fitting step.

A typical run produces edges around: `[-0.92, -0.36, -0.14, 0.05, 0.28, 0.92]`

### Train / Test Split (Quadrant Split)

States are unit-normalized random Gaussian vectors in 8D. The split is **spatial, not IID**:

- **Train**: `state[0] > 0 AND state[1] > 0` (~25% of the 8D unit sphere)
- **Test**: `state[0] < 0 AND state[1] < 0` (the opposite quadrant, disjoint)

The mean cosine between the two distributions is approximately **−0.999**, meaning train and test states point in nearly opposite directions. This is a much harder generalization challenge than a random 80/20 split, and is designed to stress-test whether each method has learned a truly generalizable correction or merely memorized training-state geometry.

### Input Features

Every method receives the same input layout:

```
X = [ state (8) | history (4 × 8 = 32) | a_curr (8) | (optional: cos_sim, band_idx) ]
```

- **state**: the current state vector (same dimensionality as action)
- **history**: 4 past bad-network actions for this state (H independent noisy forward passes through BadNet — realistic, not random-band sampled)
- **a_curr**: the current action to be corrected
- **cos_sim** *(feedback variant only)*: `cos(a_curr, GoodNet(state))` — real-time quality signal
- **band_idx** *(feedback variant only)*: `current_band / (num_bands − 1)` — normalized position in quality ladder

Base input dimension: `8 + 4×8 + 8 = 48`. Feedback variant: `50`.

### Residual Policy Architecture

All learned methods share the same network:

```
ResidualPolicy: Linear(input_dim, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, action_dim)
```

Output is a **residual vector** `r`, applied as `a_new = a_curr + r`.

### Training

- Optimizer: Adam, `lr=1e-3`, `weight_decay=1e-4`
- LR schedule: Cosine annealing from `1e-3` to `1e-4` over 500 epochs
- Batch size: 256
- Loss: MSE between predicted residual and target residual
- Metrics printed every 50 epochs for both train and test sets

---

## Methods

### Zero (Baseline Floor)

No correction applied. `r = 0`, so `a_new = a_curr = a_bad`. Measures the raw quality of BadNet's outputs as the lower bound all methods must beat.

### Direct

**The one-shot baseline.** The policy always starts from `a_curr = BadNet(state)` (the actual bad action, not a band proxy) and is trained to predict:

```
Y = GoodNet(state) − BadNet(state)
```

This is the simplest possible formulation: given the bad action and its history, directly predict the full vector that bridges it to the good action. It works well when the mapping `state → (a_good − a_bad)` is smooth enough to generalize, but it gives the policy no structural guidance about *how much* to correct.

### Band+1-Random+Full

A band-structured method with **random targets** and **full band coverage**. For each training sample:

1. Sample `current_band` uniformly from bands 0–4.
2. Rejection-sample `a_curr` in that band.
3. **If** `current_band < 4`: rejection-sample a random action in `current_band + 1` as the target.
4. **If** `current_band == 4` (top band): target is `a_good` directly.

The `+Full` suffix means the top band is covered with a direct-to-good target, eliminating the out-of-distribution problem that occurs when the policy reaches the top band during rollout.

The limitation of random targets: high variance in `Y` (any action in the next band is valid), making it hard for the network to learn a consistent function.

### Band+1-Fixed+Full

The same structure as Band+1-Random, but targets are **geometrically determined** rather than randomly sampled:

Instead of picking any action in the next band, the target is placed at the **midpoint cosine** of the next band, lying in the same great-circle plane as `(a_curr, a_good)`:

```
target_cos = (band_edges[b+1] + band_edges[b+2]) / 2

a_target = target_cos × a_good + sqrt(1 - target_cos²) × perp_unit
```

where `perp_unit` is the component of `a_curr` orthogonal to `a_good`, normalized. This gives a unique, deterministic target for every `(a_curr, a_good)` pair, dramatically reducing supervised signal variance.

Top band (band 4): target is `a_good` directly.

### Band+1-Fixed+GT+Full

The full method. Identical to Band+1-Fixed+Full, but the policy also receives two extra scalar inputs:

- `cos(a_curr, GoodNet(state))` — the policy's real-time quality signal (how aligned the current action already is)
- `current_band / 4` — normalized band index (where on the quality ladder we are)

These are recomputed at every rollout step from the actual current action, so the policy always knows its current quality position and can calibrate the size and direction of its correction accordingly.

---

## Evaluation Metrics

Every 50 epochs (train and test), and at the end of training (test only):

| Metric | Description |
|---|---|
| `MSE` | Mean squared error between predicted and target residual |
| `MeanJump` | Average change in band index after applying one residual: `mean(band(a_new) − band(a_curr))`. Ideal is **1.0** for band methods. |
| `Frac+1` | Fraction of test samples where the action moved exactly one band up. |
| `FracImpr` | Fraction where the action moved to any higher band (≥1). |
| `1-step Cos` | Mean `cos(a_curr + r_pred, a_good)` — single-step quality after correction. |
| `8x Roll Cos` | Mean cosine after applying the policy **8 times** in a closed loop, rolling the history window. The primary generalization metric. |
| `Frac≥0.99` | Fraction of test states where 8-step rollout reaches cosine ≥ 0.99 vs optimal. |

The **8-step rollout** is the key stress test: it exposes compounding errors and out-of-distribution behavior (each corrected action becomes the new `a_curr` input). A policy that merely fits training residuals but doesn't generalize will degrade quickly over 8 steps.

---

## Results (Typical Run)

```
Method                 |      MSE |  MeanJump |  Frac+1 | FracImpr | 1-step Cos | 8x Roll Cos | Frac>=0.99
------------------------------------------------------------------------------------------------------------
Zero                   |  0.25880 |     0.000 |   0.000 |    0.000 |    -0.0352 |     -0.0352 |      0.000
Direct                 |  0.00126 |     1.977 |   0.205 |    0.792 |     0.9969 |      0.9979 |      0.992
Band+1-Random+Full     |  0.19769 |     1.003 |   0.308 |    0.627 |     0.2892 |      0.8664 |      0.005
Band+1-Fixed+Full      |  0.00500 |     0.767 |   0.649 |    0.708 |     0.2881 |      0.9933 |      0.830
Band+1-Fixed+GT+Full   |  0.00094 |     0.772 |   0.772 |    0.772 |     0.3029 |      0.9974 |      0.971
```

### Key Findings

**Band+1-Random+Full** — Despite its mean band jump being nearly ideal (1.00), it achieves only 0.87 rollout cosine. High target variance during training prevents the policy from learning a consistent correction function, so errors compound rapidly during the 8-step rollout.

**Band+1-Fixed+Full** — Deterministic geometric targets collapse the training variance and produce a much stronger policy (0.99 rollout cosine). The improvement from Random → Fixed is the largest single gain in the ablation.

**Band+1-Fixed+GT+Full** — Adding the GT cosine quality feedback lowers test MSE further (0.00094 vs 0.00126 for Direct) and closes the gap to Direct on the rollout metric. At epoch 400+ it matches Direct's 0.998 rollout cosine while operating incrementally (one band per step up to band 3, then direct correction at band 4).

**Direct** — Achieves excellent rollout performance by learning the full `a_good − a_bad` function in one shot. It generalizes well even across the hard quadrant split because GoodNet/BadNet are smooth functions and their difference is learnable. Serves as the competitive upper bound.

---

## Running

```bash
pip install torch numpy
python experiment.py                              # smooth (default)
python experiment.py --gen-mode highfreq --omega 7 # high-frequency oracle
python experiment.py --gen-mode compositional      # smooth + HF blend
python experiment.py --gen-mode sparse             # deep piecewise-linear oracle
```

No other dependencies. Each run takes ~2 minutes on CPU.

### Command-Line Arguments

| Argument | Default | Description |
|---|---|---|
| `--gen-mode` | `smooth` | GoodNet architecture variant (see below) |
| `--action-dim` | 8 | State and action dimensionality |
| `--history-len` | 4 | Number of past bad actions in context |
| `--epochs` | 500 | Training epochs per method |
| `--n-train` | 8,000 | Training state count |
| `--n-test` | 2,000 | Test state count (spatially disjoint from train) |
| `--omega` | 10.0 | Sinusoidal frequency for `highfreq` / `compositional` modes |
| `--alpha` | 0.5 | HF blend weight for `compositional` mode (0 = smooth, 1 = fully HF) |
| `--seed` | 42 | Random seed |

---

## GoodNet Generation Modes

The `--gen-mode` flag controls how the frozen oracle (GoodNet) maps states to "good" actions. Each mode creates a different difficulty of generalization across the quadrant-split train/test boundary.

### `smooth` (default)

A 2-layer Tanh MLP. The `state -> a_good` function is smooth and extrapolates well across the quadrant split. All methods achieve near-perfect rollout cosine.

### `highfreq`

Random Fourier features: `normalize(MLP(sin(omega * W @ state)))`. The sinusoidal projection makes the function oscillate rapidly — at `omega=7`, the function decorrelates across the train/test boundary. Controlled by `--omega`.

### `compositional`

`normalize(smooth_MLP(state) + alpha * MLP(sin(omega * W @ state)))`. A smooth base perturbed by a high-frequency component. The smooth part extrapolates (giving directional signal); the HF part doesn't (adding magnitude noise). Controlled by `--omega` and `--alpha`.

### `sparse`

A 6-layer ReLU MLP. The many ReLU activation boundaries create a piecewise-linear function where different linear regions activate in the training vs. test quadrant. Despite being continuous, the function can change character across the split.

### How Modes Affect Extrapolation

The diagnostic `cos(a_good_train, a_good_test)` (printed during setup) measures how correlated good actions are across the split:

| Mode | Typical cross-state cos | Direct rollout | B1-GT+Full rollout |
|---|---|---|---|
| `smooth` | ~1.0 | 0.998 | 0.998 |
| `sparse` | ~1.0 | 1.000 | 1.000 |
| `highfreq --omega 5` | 0.42 | 0.920 | 0.905 |
| `compositional --omega 8 --alpha 0.4` | 0.34 | 0.918 | 0.892 |
| `highfreq --omega 7` | 0.28 | 0.811 | 0.704 |
| `highfreq --omega 8` | 0.22 | 0.711 | 0.599 |
| `highfreq --omega 9` | 0.18 | 0.577 | 0.493 |
| `compositional --omega 15 --alpha 0.8` | 0.22 | 0.393 | 0.383 |

### Key Observations

**When the function is extrapolatable** (`smooth`, `sparse`): Both Direct and Band+1-GT+Full achieve near-perfect rollout. The 6-layer ReLU `sparse` net turned out to be surprisingly smooth across the quadrant split (cross-state cos = 1.0), so it doesn't differentiate the methods.

**When the function partially extrapolates** (`highfreq --omega 5-7`, `compositional`): Direct degrades but remains ahead because its single-shot prediction has no compounding error. Band+1 has the smaller per-step MSE generalization gap (e.g., 7.4x vs 14.5x at `--omega 5`) but accumulates errors over 8 rollout steps.

**When the function doesn't extrapolate** (`highfreq --omega 9+`, `compositional --omega 15 --alpha 0.8`): All methods collapse toward the zero baseline. Neither Direct nor Band+1 can extrapolate the correction direction across the split.

**Why Direct holds up**: In this synthetic setting, Direct makes one prediction from `(state, history, a_bad)` with a single error. Band+1 makes 8 predictions that each carry extrapolation error, and these errors compound through the rollout. The GT cosine feedback tells the policy *how far* from optimal it is but not *which direction to go* — the direction still requires state-dependent function extrapolation that fails equally for both approaches.

**Where Band+1 would win**: In environments with sequential state dynamics (where the optimal action depends on past corrections), non-stationary targets, or settings where the rollout feedback provides actionable directional information (e.g., if the policy were trained on multi-step correction trajectories via DAgger). The current synthetic setup is fundamentally a static function approximation problem where single-shot prediction has a structural advantage.

---

## File Structure

```
experiment.py   Single-file experiment — all code, data generation, training, and evaluation
README.md       This file
```

### Code Sections

| Section | What it does |
|---|---|
| `GoodNet`, `BadNet` | Frozen RND-style oracle and bad policy networks |
| `HighFreqGoodNet`, `CompositionalGoodNet`, `SparseGoodNet` | Alternative oracle architectures for extrapolation experiments |
| `make_good_net` | Factory that selects GoodNet variant by `--gen-mode` |
| `fit_band_edges` | Fits 5-quantile band boundaries from training-state cosine similarities |
| `fixed_band_target` | Geometric interpolation to place a target at an exact cosine value in the `(a_curr, a_good)` plane |
| `generate_states_split` | Quadrant spatial split of the unit sphere |
| `generate_dataset` | Builds `(X, Y_residual)` tensors for any method; history always sampled from BadNet |
| `eval_policy` | Single-step band metrics + optional multi-step rollout |
| `_rollout_eval` | Closed-loop 8-step rollout with rolling history and optional GT feedback |
| `ResidualPolicy` | 3-layer ReLU MLP (hidden=256), shared architecture across all methods |
| `train_policy` | Training loop with cosine LR schedule; prints metrics every 50 epochs |
