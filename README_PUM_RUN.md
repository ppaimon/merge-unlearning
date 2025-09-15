# PUM: Perturb–Unlearn–Merge (How to Run)

This document explains how to run the PUM trainer in this repo and how its knobs map to the methodology. The implementation is dataset‑agnostic; the provided `scripts/tofu_pum.sh` mirrors the baseline flow used by `scripts/tofu_unlearn.sh` for fair comparison on TOFU.

## Quick Start

- Train + evaluate PUM using defaults:
  - `bash scripts/tofu_pum.sh`
- Baseline for comparison:
  - `bash scripts/tofu_unlearn.sh`

The PUM trainer itself makes no TOFU‑specific assumptions; only the script uses TOFU paths/configs. Swap out the script’s model, datasets and experiment config for other tasks without code changes.

## Methodology Mapping (high‑level)

- Zero‑sum correlated base noises with secret scalings: per copy `k` and layer/param, draw base zero‑sum Gaussians and scale by private `α_k ≥ 1`.
- Reparameterization and publication: optional per‑copy orthogonal/permutation transforms inside attention/FFN (server records/inverts them). Server publishes clipped center `\~θ` + noise + tiny jitter `ξ_k`.
- Client‑side local unlearning: run a small‑step inner method (e.g., GradAscent/GradDiff/NPO) per copy starting from the published parameters; no client clipping.
- Inverse alignment and harmonic‑normalized denoising: invert transforms and take a harmonic‑normalized mean with weights `w_k ∝ 1/α_k`.
- Global update (streaming): maintain `S0 = Σ 1/α_k`, `S1 = Σ (1/α_k) Δ_k`, then apply `θ ← θ + η_srv · (S1/S0)` without storing all updates.

## Key Features in Code

- Streaming aggregation to save memory: exact result of storing all `m` updates with only `O(d)` memory.
- DP‑aware publication center with EMA reference and clipping: deterministic reference `θ_ref` updated per round, with per‑layer/global thresholds to bound sensitivity.
- Per‑layer DP noise calibration with “choose‑smaller‑noise” policy: for each RDP order, compute both equalized and variance‑minimizing schedules and choose the smaller total variance. Falls back to single‑σ when per‑layer inputs are unavailable.
- Optional tiny jitter `τ` per copy: helps ensure full‑rank joint covariance; negligible for RDP at leading order.
- Optional reparameterization: orthogonal invariance in attention, permutations in FFN hidden channels; inverted on server for alignment.

## Hyperparameter Quick Reference (mirrors `scripts/tofu_pum.sh`)

- Outer loop
  - `copies_m` (`PUM_COPIES_M`): number of perturbed copies per round `m` (4–16).
  - `rounds_R` (`PUM_ROUNDS_R`): outer rounds `R` (1–3).
  - `alpha_min`, `alpha_max` (`PUM_ALPHA_MIN/MAX`): secret scalings range, `α_k ∈ [α_min, α_max]`, `α_min ≥ 1` (1.0–1.5).
  - `eta_srv` (`PUM_ETA_SRV`): server step size `η_srv` (0.5–1.5).
- Publication center, EMA, jitter
  - `theta_ref_beta` (`PUM_THETA_REF_BETA`): EMA weight for `θ_ref` (0.7–0.9).
  - `server_center_clipping` (`PUM_SERVER_CENTER_CLIP`): enable center clipping; `null` auto‑enables if sensitivities or `C` provided.
  - `center_clip_C_per_layer` (`PUM_CENTER_CLIP_C_PER_LAYER`): per‑layer `C_ℓ` so that `Δ̄_{2,ℓ} = 2 C_ℓ`.
  - `center_clip_C_global` (`PUM_CENTER_CLIP_C_GLOBAL`): global `C` for non‑layer params.
  - `jitter_tau` (`PUM_JITTER_TAU`): tiny jitter std `τ` per copy (0–1e‑3).
- DP calibration
  - `per_layer_noise` (`PUM_PER_LAYER_NOISE`): calibrate per‑layer `σ_ℓ` if true; else a single `σ`.
  - `dp_epsilon`, `dp_delta` (`DP_EPSILON/DELTA`): DP target budget.
  - `dp_rdp_orders` (`DP_RDP_ORDERS`): RDP orders sweep.
  - `dp_per_layer_allocation` (`DP_PER_LAYER_ALLOC`): `auto` (choose smaller variance), `equalized`, or `varmin`.
  - Sensitivity inputs: prefer `dp_sensitivity_per_layer_l2` (`DP_SENS_PER_LAYER_L2`); else `dp_sensitivity_total_l2` (`DP_SENS_TOTAL_L2`). If `per_layer_noise=true` but no per‑layer sensitivity is provided, the trainer falls back to single‑σ.
  - `dp_use_worstcase_alpha` (`DP_USE_WORSTCASE_ALPHA`): if true, uses `S_α = m/α_min^2` for DP; otherwise, uses a uniform expectation over `[α_min, α_max]`.
- Manual noise (discouraged when DP is given)
  - `sigma` (`PUM_SIGMA`): global noise if DP unset.
  - `sigma_per_layer` (`PUM_SIGMA_PER_LAYER`): per‑layer noise if DP unset.
- Inner unlearning and robustness
  - `inner_handler`: one of GradAscent, GradDiff, NPO, DPO, etc.
  - `local_epochs` (`PUM_LOCAL_EPOCHS`): per‑copy training epochs (1–3).
  - `local_max_steps` (`PUM_LOCAL_MAX_STEPS`): steps cap per copy; `null` enables auto‑balancing vs. rounds.
  - `clip_update_norm` (`PUM_CLIP_UPDATE_NORM`): global L2 clip on aligned update.
  - `clip_update_norm_per_layer` (`PUM_CLIP_UPDATE_PER_LAYER`): per‑layer L2 clips.
- Reparameterization
  - `use_orthogonal_reparam` (`PUM_USE_REPARAM`): enable per‑copy orthogonal/permutation transforms.

## Running on Other Datasets

- Replace dataset/model entries and experiment config in `scripts/tofu_pum.sh`; the trainer code is dataset‑agnostic.
- Provide public‑only sensitivity proxies if you want DP per‑layer calibration (e.g., from quantiles/heuristics of public checkpoints).

## Notes on Memory and Efficiency

- Streaming aggregation keeps only `S0` (scalar) and `S1` (one model‑shaped accumulator) in memory; no need to store all `m` client updates.
- Optional per‑copy transforms and jitter add negligible overhead relative to local unlearning.

## Outputs and Checkpoints

- PUM creates per‑copy subfolders under `saves/unlearn/<task>/pum_round{r}_copy{k}/` for inner runs, without saving checkpoints (inner `save_strategy=no`).
- The final merged model is saved to `saves/unlearn/<task>` like the baseline for head‑to‑head evaluation.

## Common Recipes

- Utility‑oriented with DP off:
  - Leave `DP_EPSILON/DELTA=null` and set a small `PUM_SIGMA` (or `PUM_SIGMA_PER_LAYER`).
- DP on, per‑layer calibration:
  - Set `PUM_PER_LAYER_NOISE=true`, provide `DP_EPSILON`, `DP_DELTA`, and per‑layer sensitivities via `DP_SENS_PER_LAYER_L2` (preferred), or a total via `DP_SENS_TOTAL_L2`.
  - Keep `DP_PER_LAYER_ALLOC=auto` to apply “choose smaller noise” policy.
- Stronger obfuscation:
  - Set `PUM_USE_REPARAM=true` and widen `α_max` moderately (e.g., 1.2–1.4), balancing utility vs. obfuscation.

## Troubleshooting

- If per‑layer DP is requested but the model’s number of layers cannot be determined, the trainer falls back to single‑σ DP calibration and logs a warning.
- If no feasible DP calibration is found for given `(ε, δ)`, the trainer keeps the existing manual `σ` settings and logs a warning.

---
If you need additional examples or a ready‑made template for non‑TOFU datasets, open an issue or ask to add a dataset‑agnostic run script stub.
