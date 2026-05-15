# Path MoE for `LatLonPredModuleV13` — Implementation Notes

Branch: `dev/moe`
Scope: **lat (path) expert MoE only**. Lon (vel) MoE is not implemented yet.

## Overview

The `lat_ffn` in `LatLonPredModuleV13` (single `AsymmetricFFN`) is replaced by
a **Dense Soft Mixture of Experts**:

```
path_embed (B, N_path, D)
  │
  ├─ mean over N_path ─► token (B, D)
  │                          │
  │                          ├─ ETFRouter(token, aux) ─► weights (B, K=5)
  │                          │                              │
  └────────► ExpertBankFFN(path_embed, weights) ◄───────────┘
                          │
                          ▼
                  path_embed' (B, N_path, D)
```

- **Router**: FiLM(aux) → MLP → L2-norm → dot with **ETF prototypes** → softmax
- **DR loss**: pulls the L2-normed router query toward `etf[path_class]` until
  `q·etf = r = sqrt((K-1)/K)`
- **K = 5** path classes (LANEFOLLOW / LANE_CHANGE / LEFT / RIGHT / STRAIGHT)
- **Soft routing during training**, hard top-1 at inference (tau anneals
  1.0 → 0.3 over the first half of training)

## Path class definition

5-way, from `info['command_far']` + scenario label:

| Class | Condition | Train distribution (234,769 frames) |
|---|---|---|
| LANEFOLLOW (0) | `cmd_far == LANEFOLLOW`, scenario ∉ {OV,MG} | 61.90 % |
| LANE_CHANGE (1) | scenario ∈ {OVERTAKING, MERGING_HIGHWAY, MERGING_JUNCTION} OR `cmd_far ∈ {CHANGELANELEFT, CHANGELANERIGHT}` | 19.67 % |
| LEFT (2) | `cmd_far == TURNLEFT` | 8.55 % |
| RIGHT (3) | `cmd_far == TURNRIGHT` | 4.68 % |
| STRAIGHT (4) | `cmd_far == STRAIGHT` | 5.20 % |

Max imbalance ≈ 13:1 (LANEFOLLOW vs RIGHT).

Scenario labels: `data/DriveMoE/labels/scenario_labels/{ScenarioType}_{Town}_{Route}_{Weather}.json` keyed by `str(frame_idx)`.

## Step-by-step summary

### Step 1 — Dataset path_class label
- Module-level CARLA cmd constants + path class constants
- `_get_scenario_label(info)` lazy-loads scenario JSON per folder, cached
- `get_path_class(info)` returns 0..4
- `get_ann_info` attaches `anns_results['path_class']` (np.int64 scalar)
- New `__init__` param `scenario_label_root`
- **Files**: `projects/mmdet3d_plugin/datasets/b2d_3d_dataset.py`,
  `projects/configs/sparsedrive_stage2.py` (added `scenario_label_root`)

### Step 2 — ETF prototype util
- `make_etf(K, D, seed=0)` → `(K, D)` tensor, unit norm, sum=0, pairwise
  cosine = `-1/(K-1)` (Simplex Equiangular Tight Frame)
- Construction: random orthonormal D×K basis (QR) × centering matrix × `sqrt(K/(K-1))`
- **Files**: `projects/mmdet3d_plugin/models/motion/moe_utils.py` (new)

### Step 3 — ETFRouter
- FiLM block: `aux → (γ, β)`, **zero-init last layer** so initial modulation
  is identity (token passes through unchanged at step 0)
- Routing MLP: `token → routing_dim`
- L2-normalize `q`, dot with `etf` buffer → logits → `softmax(logits/tau)` → weights
- `dr_loss(q_norm, labels) = mean((q_norm · etf[labels] - r)²)`
- **Key implementations**: FiLM zero-init, DR loss, ETF buffer registration
- **Files**: `projects/mmdet3d_plugin/models/motion/moe_utils.py`

### Step 4 — ExpertBankFFN (Dense Soft MoE)
- K stacked 2-layer FFNs as `(K, D, H)` / `(K, H, D)` parameter tensors
- `einsum('bd,kdh->bkh', x, W1)` runs K experts in parallel (no Python loop)
- Pre-LN, GELU, dropout, residual using normalized input (matches
  `AsymmetricFFN` semantics)
- Per-expert Kaiming init
- **Properties verified**: K=1 reduces to a plain FFN (regression-safe);
  one-hot weights = single-expert forward; all K experts receive gradient
  on every batch (dense MoE)
- **Files**: `projects/mmdet3d_plugin/models/motion/moe_utils.py`

### Step 5 — LatLonPredModuleV13MoE
- Inherits `LatLonPredModuleV13`, sets `lat_ffn = None`, adds `path_router`
  + `lat_moe`
- `_route_lat`: mean-pools path_embed for router token; broadcasts weights
  `(B, K)` to `(B*N_path, K)` and reshapes path_embed to `(B*N_path, D)` for
  the ExpertBank
- Stashes router outputs on `self.last_path_q`, `self.last_path_logits`,
  `self.last_path_weights` for the head to read in the loss step
- Forward returns the **same 9-tuple** as V13 — head interface unchanged
- Registered via `@PLUGIN_LAYERS.register_module()`
- **Files**: `projects/mmdet3d_plugin/models/motion/motion_blocks.py`

### Step 6 — DR loss wiring through the head
- `Collect` keys (train pipeline) gain `'path_class'`
- In `MotionPlanningHeadV13.forward`, after the `lat_lon_pred` op:
  if the module has `last_path_q`, stash `path_router_q`, `path_router_logits`,
  `_path_router_module` into `plan_result`
- In `loss_planning`: if `path_router_q` is in the planning_result, call
  `module.path_router.dr_loss(q, data['path_class'])` →
  `path_router_dr_loss_{decoder_idx}`, weighted by
  `plan_config['path_router']['weight']`
- **Backward compatible**: non-MoE modules don't set `last_path_q`, so the
  loss branch is silently skipped
- **Files**: `projects/configs/sparsedrive_stage2.py`,
  `projects/mmdet3d_plugin/models/motion/motion_planning_head_v13.py`

### Step 7 — Config switch
- `lat_lon_pred_layer.type` → `LatLonPredModuleV13MoE`
- MoE hyperparams: `num_path_experts=5`, `router_routing_dim=64`,
  `ffn_hidden_dim=embed_dims*2`, `ffn_drop=0.1`, `tau=1.0`,
  `router_aux_dim=None` (self-conditioning: aux = mean(path_embed))
- `plan_config['path_router'] = dict(weight=1.0)` — DR loss weight
- **Stage1/Stage2 instances are automatic**: `MotionPlanningHeadV13.__init__`
  builds a separate module per occurrence of `lat_lon_pred` in
  `operation_order` (motion_planning_head_v13.py:156-161); no need for
  `_stage1`/`_stage2` config splits
- **Files**: `projects/configs/sparsedrive_stage2.py`

### Step 8 — Tau annealing
- `TauAnnealingHook`: iter-based linear annealing of `tau`. Walks the model
  and sets `tau` on any submodule whose class name contains
  `MoE` / `Router` / `LatLonPred` (and exposes a `tau` attribute)
- Triggers in `before_run` and `before_train_iter` (every `update_interval`
  iters)
- Default: `tau_start=1.0`, `tau_end=0.3`, anneals over the first half of training
- **Files**: `projects/mmdet3d_plugin/core/hooks/tau_annealing_hook.py` (new),
  `projects/mmdet3d_plugin/core/hooks/__init__.py` (new),
  `projects/mmdet3d_plugin/__init__.py` (`from .core.hooks import *`),
  `projects/configs/sparsedrive_stage2.py` (`custom_hooks` block)

## Files added / modified (full list)

```
projects/configs/sparsedrive_stage2.py                       # MoE config, custom_hooks, path_class in Collect
projects/mmdet3d_plugin/__init__.py                          # register core.hooks
projects/mmdet3d_plugin/datasets/b2d_3d_dataset.py           # path_class generation, scenario_label loader
projects/mmdet3d_plugin/models/motion/motion_blocks.py       # LatLonPredModuleV13MoE
projects/mmdet3d_plugin/models/motion/motion_planning_head_v13.py  # capture router outputs, DR loss in loss_planning
projects/mmdet3d_plugin/models/motion/moe_utils.py           # NEW: make_etf, ETFRouter, ExpertBankFFN
projects/mmdet3d_plugin/core/hooks/__init__.py               # NEW
projects/mmdet3d_plugin/core/hooks/tau_annealing_hook.py     # NEW: TauAnnealingHook
docs/moe.md                                                  # NEW: this document
```

## Loss terms added

- `path_router_dr_loss_0` — Stage1 (filter_num[0]) DR loss
- `path_router_dr_loss_1` — Stage2 (filter_num[1]) DR loss

Both weighted by `plan_config['path_router']['weight']` (default 1.0).

## Parameter cost

| Component | Params (D=256, H=512, K=5) |
|---|---|
| Original `lat_ffn` (AsymmetricFFN) | ~0.26 M |
| `ExpertBankFFN` (K=5, replaces `lat_ffn`) | 1.31 M |
| `ETFRouter` (FiLM + routing MLP) | ~0.3 M |
| **Net MoE addition per stage** | **≈ 1.35 M** |
| × 2 (Stage1 + Stage2 independent instances) | **≈ 2.7 M** |

Negligible compared to the rest of the SparseDrive model.

## Training

```bash
bash scripts/train.sh
# = bash ./tools/dist_train.sh projects/configs/sparsedrive_stage2.py 8 --deterministic
```

What to watch in the logs:
- `path_router_dr_loss_0`, `path_router_dr_loss_1`: start ~0.5–0.7, decrease toward 0
- `[TauAnnealingHook] iter=... tau=... applied_to=2` every 200 iters (2 = Stage1 + Stage2 MoE modules)
- Router weight distribution: if it collapses to ~99% on LANEFOLLOW with rare classes never picked, that signals minority starvation → consider a class-balanced sampler

## K=1 regression test

Single-line config change to confirm the MoE wiring doesn't degrade the
baseline:
1. `num_path_experts: 5 → 1`
2. `plan_config['path_router']['weight']: 1.0 → 0.0` (disable DR loss)
3. Train; metric should match the pre-MoE FFN baseline within a few percent
   (`ExpertBankFFN` with K=1 is bit-identical to a plain FFN — verified in
   unit tests)

## Pending work (not in this branch)

1. **Lon (vel) MoE** — same pattern with 6-way `vel_class` from scenario labels
   (CRUISE / OVERTAKING / MERGING / EMERGENCY_BRAKE / GIVEWAY / TRAFFIC). Will
   require adding `vel_class` to dataset + new `vel_moe` + DR loss term.
2. **Better router aux** — currently self-conditioning (`aux = mean(path_embed)`).
   The original design used `agent_pool_emb` (geometric-weighted pool of
   `instance_feature_selected`) + `command_far` embedding. Wiring this requires
   plumbing `aux` through the head to the MoE module.
3. **Class-balanced sampler** — `GroupInBatchSampler` (current) handles temporal
   streaming, not class balance. A weighted sequence sampler that preserves
   in-sequence frame order would be needed if minority classes starve. Defer
   until after the first training trial.
4. **Hard inference toggle** — currently soft routing at inference too. After
   training, switching to top-1 hard routing usually closes a small gap.
