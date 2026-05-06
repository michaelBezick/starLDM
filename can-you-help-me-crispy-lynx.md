# Plan: Selector-Guided Latent Diffusion Planning with Branch Repulsion

> Saved as `plan.md` at the repo root after approval.

## Context

STAR-LDM already runs a latent diffusion "thinking" phase (`TransfusionGPT.sample` → `diffusion_model_predictions`) before GPT-2 decodes. The repo also ships a frozen-conditioning classifier guidance path: `NoiseConditionedMLP` (`star_ldm/models/classifier.py`) is hooked at `star_ldm/models/transfusion.py:447-463` to drive `grad_{z_t} BCE` into `pred_eps`.

We want a first-pass research prototype that:

1. Trains a **supervised selector** `s_eta(prompt, z_t, t) → logit` whose sigmoid approximates `P(decoded answer is correct | prompt, z_t, t)`.
2. Uses `grad_{z_t} log sigmoid(s_eta)` as an inference-time **steerer** of the diffusion trajectory — the same hook the existing classifier uses.
3. Optionally runs **K parallel diffusion branches** per prompt with a **repulsion** force between them, then picks the highest-scoring final plan.

Goal: reuse the existing model loader, sentence encoder, noise schedule, classifier-guidance gradient pattern, and EMA-checkpoint conventions. **No invasive rewrites.** No learned steerer or RL yet.

---

## Architectural decisions (motivated by code inspection)

- **Prompt embedding for the selector** = the existing frozen Sentence-T5 XL encoder already inside `TransfusionGPT` (`transfusion.py:148-153`). Selector takes a 768-dim prompt embedding (encoded once per prompt, reused across all `t` and all K branches). Simple, no GPT-2 hidden-state plumbing.
- **Latent the selector sees** = the **raw internal normalized diffusion latent** that flows through the reverse loop, *before* any `sample()` post-processing. `transfusion.py:309-311` currently does `unnormalize_sentence_emb` + `F.normalize(...)` on `x_start` before returning it; that post-processed vector is the wrong target. The selector trains and is queried on the loop-internal `x_start` / `z_t` (which is in the same normalized space the diffusion model itself operates in). To make this clean we expose the internal latent explicitly (see §3 and §5c).
- **Selector architecture** = mirror `NoiseConditionedMLP` but condition on the prompt embedding via simple concatenation (`Linear([z_t || prompt_emb] → mlp_dim)`) before the time-conditioned MLP stack. Reuses `ConditionableMLP`, `SinusoidalPosEmb`, `FeedForward` from `star_ldm/models/modules/`.
- **Guidance injection point** = mirror the classifier branch in `diffusion_model_predictions` (`transfusion.py:447-463`). Add a parallel selector branch with the same `with torch.enable_grad(); z_t.requires_grad=True; grad = torch.autograd.grad(loss, z_t)` shape, then `pred_eps += scale * sigma2.sqrt() * grad`. Selector loss is BCE-toward-1 (gradient descent on this loss = gradient ascent on `log P(correct)`); repulsion is a **redundancy loss** (high when branches are similar, low when diverse) so its gradient *also descends* — same sign convention as the classifier path. Backwards compat: when `selector_guidance == 0` and no `selector` is passed, behavior is byte-identical to today.
- **K branches** = duplicate `input_ids` K times inside `TransfusionGPT.sample` (cheap: only the prompt tokens are repeated; the diffusion is already batched). Repulsion becomes a `(B, K, D)` reshape on `z_t` — *not* on a detached `pred_x` — inside the guidance hook so autograd can actually flow.
- **Selector module location** = `star_ldm/selector.py` (per spec; classifier lives at `models/classifier.py` so this stays parallel and findable).
- **Checkpoint format** for the selector = follow the EMA convention used by `load_classifier` (`classifier.py:132-188`): `{'ema': ..., 'model': ..., 'opt': ..., 'step': ...}` + sibling `config.yaml` so we can lean on the `load_classifier`-style loader.

---

## Files to add

| File | Purpose |
|---|---|
| `star_ldm/selector.py` | `Selector` model + `load_selector(path, device)` |
| `star_ldm/verification/__init__.py` | export `BaseVerifier`, `ExactMatchVerifier`, `get_verifier(name)` |
| `star_ldm/verification/base.py` | `BaseVerifier.verify(prompt, decoded, gold) -> bool` |
| `star_ldm/verification/exact_match.py` | normalized exact-match verifier |
| `scripts/collect_selector_data.py` | sample plans from frozen STAR-LDM, decode, verify, dump dataset |
| `scripts/train_selector.py` | BCE + optional pairwise rank loss, EMA, WSD/cosine schedule |
| `configs/selector_train.yaml` | example selector training config |
| `configs/generate_with_selector.yaml` | example single-branch selector-guided generation |
| `configs/generate_kbranch_repulsion.yaml` | example K-branch + repulsion |
| `tests/test_selector_unit.py` | fast CPU unit tests (no model checkpoint needed) |
| `tests/test_selector_integration.py` | full STAR-LDM generation tests, **opt-in** via marker / env var |
| `tests/conftest.py` | shared fixtures + `--run-integration` flag |

## Files to modify (minimal, additive only)

| File | Change |
|---|---|
| `star_ldm/models/transfusion.py` | extend `diffusion_model_predictions` and `sample` with `selector`, `selector_kwargs` (scale, schedule window, clip, normalize, repulsion, K). Expose internal normalized latent. Keep `cls_guidance` path untouched. |
| `star_ldm/interface.py` | optional `selector_path` arg in `__init__`; `generate(...)` forwards new kwargs through to `sample` and returns an explicit dataclass. |
| `scripts/generate.py` | new CLI flags below; pass through. |
| `README.md` | short "Selector-guided generation" section with example commands. |

---

## 1. Selector model — `star_ldm/selector.py`

```python
class Selector(nn.Module):
    def __init__(self, sentence_emb_dim=768, mlp_dim=768, mlp_hidden_dim=1536,
                 mlp_depth=4, prompt_emb_dim=768, global_norm=True,
                 dataset_name='fineweb_100b'):
        # time_mlp = SinusoidalPosEmb -> Linear -> SiLU -> Linear  (matches NoiseConditionedMLP)
        # input_proj = Linear(z_dim + prompt_dim, mlp_dim)
        # body      = ConditionableMLP(input_dim=mlp_dim, ..., n_layers=mlp_depth)  -> Linear(mlp_dim, 1)
        # data_mean / data_std loaded from DATA_STATS_PATH; selector expects ALREADY-NORMALIZED latents
        # (i.e. the loop-internal representation) and does NOT renormalize at forward time.
```

API:

```python
selector(prompt_embeds, z_t, alpha2)            # supports (B, D) or (B*K, D); we always flatten K up front
selector.get_logits(z_t, alpha2, prompt_embeds) # -> (B, 1)
selector.get_score_loss(z_t, alpha2, prompt_embeds, target=1.0)  # BCE-toward-1 for guidance
selector.get_loss(z_t, alpha2, prompt_embeds, labels)            # BCE for training
```

**Important contract:** `z_t` arguments are in the **internal normalized** space — same space as the input to `diffusion_model_predictions` and the output of the loop's `x_start` prior to `sample()` post-processing. Callers that have an *unnormalized* sentence embedding must call `model.normalize_sentence_emb(...)` first.

`load_selector(path, device)` mirrors `load_classifier` (`star_ldm/models/classifier.py:132`) — reads `config.yaml`, prefers `best_model.pt`, extracts EMA weights via `ema_pytorch.EMA` if present.

**Reuse:** `SinusoidalPosEmb` (`models/modules/diffusion.py`), `FeedForward` + `ConditionableMLP` pattern (`classifier.py:13-37`), `DATA_STATS_PATH` (`data/CONSTANTS.py`).

---

## 2. Verifier interface — `star_ldm/verification/`

```python
class BaseVerifier:
    def verify(self, prompt: str, decoded: str, gold: Optional[str] = None) -> bool: ...

class ExactMatchVerifier(BaseVerifier):
    # Normalize whitespace + lowercase + strip; compare to gold
```

`get_verifier(name)` registry so GSM8K/MATH/code verifiers can be added later without touching the data-collection script.

---

## 3. Selector data collection — `scripts/collect_selector_data.py`

Inputs:
- `--model_path` STAR-LDM checkpoint dir
- `--prompts_path` JSONL: `{"prompt_id", "prompt", "gold"}`
- `--output_path` `.pt`
- `--num_samples_per_prompt N`  (default 16)
- `--verifier exact_match`  (registry lookup)
- `--save_noisy_timesteps`  (bool — also save corrupted z_t / t pairs)
- `--noisy_per_clean K_t`  (default 4)
- `--sampling_timesteps`, `--sampler`, `--cosine_scale`, etc. (forwarded to `model.sample`)
- `--device`, `--seed`

Procedure:

1. Load the frozen `TransfusionGPTInterface(model_path)`. Encode every prompt once with `model.get_sentence_embedding(prompt)` → `prompt_emb` (768-dim, already normalized via `normalize_sentence_emb` per `transfusion.py:472-473`).
2. For each prompt, run `model.sample(input_ids, ..., return_internal_latents=True)`. We extend `sample()` (§5c) to also return the **loop-internal normalized x_start** — i.e., the `x_start` value at the end of the reverse loop *before* the lines at `transfusion.py:309-311` apply `unnormalize_sentence_emb` + `F.normalize`. The selector trains on this representation.
3. Decode deterministically (greedy: `do_sample=False, num_beams=1`).
4. `r = verifier.verify(prompt, decoded, gold)` → 0/1 label.
5. If `--save_noisy_timesteps`, sample `K_t` timesteps `t ~ U[0, 1]` per clean latent, compute `alpha2 = get_scaled_noise_schedule('cosine', scale=cosine_scale)(t)`, draw `eps ~ N(0, I)`, build `z_t = sqrt(alpha2)*z_0 + sqrt(1-alpha2)*eps` (use `variance_preserving_map` from `transfusion.py:32`). All in the same normalized space.
6. Save as a single `.pt` with tensors and a list of metadata dicts:
    ```
    {
      'prompt_ids':  LongTensor[N_total],
      'prompt_embs': FloatTensor[N_total, 768],
      'z0_internal': FloatTensor[N_total, 768],   # loop-internal normalized latent
      'zt':          FloatTensor[N_total, K_t, 768] | None,
      'alpha2':      FloatTensor[N_total, K_t]      | None,
      'labels':      FloatTensor[N_total],
      'meta':        [{'prompt_id', 'prompt', 'decoded', 'gold'}, ...],
    }
    ```
   `pt` is the right choice here — these are mostly tensors and we want a single file.

**Reuse:** `TransfusionGPTInterface._load_model` (full model loading), `TransfusionGPT.get_sentence_embedding` (`transfusion.py:467`), `variance_preserving_map`, `get_scaled_noise_schedule` (`diffusion/noise_schedule.py`).

---

## 4. Selector training — `scripts/train_selector.py`

CLI flags (per spec):
`--data_path --output_dir --batch_size --lr --epochs|--num_steps --lambda_rank --use_timestep_selector --hidden_dim --num_layers --seed --device --wandb`

Loss:
- BCE: `BCEWithLogitsLoss(logit, r)` over the batch.
- Optional pairwise rank loss within prompt: for each prompt with at least one positive and one negative in the batch, sample one (pos, neg) pair → `-log sigmoid(s_pos - s_neg)`. Sum and average.
- `loss = bce + lambda_rank * rank_loss`.

When `--use_timestep_selector`:
- Each example is `(prompt_emb, z_t, alpha2, label)`. Either pre-saved (from `--save_noisy_timesteps`) or generated on the fly per epoch by re-noising `z0_internal`.
Otherwise: `(prompt_emb, z0_internal, alpha2=1.0, label)` (clean latent, alpha2≈1 → time_emb is a constant; selector still works).

All latents in the dataset are already in the loop-internal normalized space (§3.2), so the trainer feeds them directly to `selector(...)` without renormalization.

Training loop (lightweight, no Accelerate to keep selector independent and the script small):
- Optimizer: `get_adamw_optimizer` from `star_ldm/training/trainer.py:57` (it splits weight-decayable params correctly).
- LR schedule: `get_linear_wsd_schedule` from `trainer.py:90` (mirror main trainer) or HF cosine.
- EMA: `ema-pytorch` (same library/options as classifier and main model).
- Save `{output_dir}/model.pt`, `{output_dir}/best_model.pt` (best val AUROC), `{output_dir}/config.yaml` (so `load_selector` works the same way as `load_classifier`).

Logging (stdout always, wandb optional, off by default):
- `bce_loss`, `rank_loss`, `accuracy`, `auroc` (sklearn — guard import), `within_prompt_rank_acc` (fraction of pairs where pos > neg).

**Frozen-param guarantee:** the selector trainer never imports `TransfusionGPT`. Prompts arrive as pre-computed embeddings in the dataset file. Test enforces this (see §7).

---

## 5. Selector-guided sampling — modify `transfusion.py` + `interface.py` + `scripts/generate.py`

### 5a. New CLI flags (`scripts/generate.py`)

Per spec, all default to disabled:
```
--selector_path                str   default None
--selector_guidance            float default 0.0
--selector_guidance_start      float default 0.0
--selector_guidance_end        float default 1.0
--guidance_clip                float default 1.0
--normalize_guidance_grad      flag  default False
--num_plan_branches            int   default 1
--repulsion_scale              float default 0.0
--repulsion_metric             str   default 'cosine'   choices [cosine, l2]
--repulsion_start              float default 0.0
--repulsion_end                float default 0.7
--quality_weighted_repulsion   flag  default False
--select_best_plan             flag  default False
--save_plan_stats              str   default None       # path to JSONL/JSON output
```

(No `--repulsion_use_x0_hat` flag: repulsion is computed on `z_t` only — see 5b.)

### 5b. Hook in `TransfusionGPT.diffusion_model_predictions`

Add a *parallel* branch to the existing `cls_guidance` block. Two key invariants:

1. **Sign convention matches the classifier path.** The classifier code adds `+cls_guidance * sigma2.sqrt() * grad` to `pred_eps` where `grad = ∇_{z_t} BCE(target)`. We follow the same sign for both selector and repulsion: define a *loss* whose gradient should descend, and add it to `pred_eps` with the *same positive scale*. For the selector this is `BCE(logit, target=1.0)` (descending it = increasing P(correct)). For repulsion, define a **redundancy loss** `R_red = +sum_{i<j} cos(z_i, z_j)` (or `+sum_{i<j} exp(-||z_i-z_j||^2 / tau)`): high when branches are similar, low when diverse. Descending `R_red` pushes branches apart — exactly the desired behavior, and the same sign as the classifier and selector terms. **No flipped signs anywhere.**
2. **Repulsion gradient flows through `z_t` directly.** We do **not** detach `pred_x` and call `autograd.grad` on it — that gives a zero gradient. Repulsion is built from `z_t` (which has `requires_grad=True` inside the `enable_grad` block) so `autograd.grad(R_red, z_t)` is well-defined.

```python
if selector is not None and selector_guidance != 0.0 and t_in_window(t, sg_start, sg_end):
    sigma2 = 1 - alpha2
    with torch.enable_grad():
        z_t.requires_grad_(True)

        # --- Selector term ---
        # BCE-toward-1: minimizing it == maximizing log sigmoid(logit) == maximizing P(correct).
        sel_loss = selector.get_score_loss(z_t, alpha2, prompt_embeds).sum()

        # --- Repulsion term (redundancy loss, computed directly on z_t) ---
        if repulsion_scale != 0.0 and t_in_window(t, rep_start, rep_end) and K > 1:
            zBKD = z_t.view(B, K, D)                                       # NOT detached.
            if metric == 'cosine':
                zn = F.normalize(zBKD, dim=-1)
                # Redundancy = sum of pairwise cosine sims. Higher = more redundant = larger loss.
                gram = einsum('b k d, b j d -> b k j', zn, zn)
                upper = torch.triu(gram, diagonal=1)
                R_red = upper.sum()
            elif metric == 'l2':
                # Redundancy via RBF: high when close. tau is a config constant.
                d2 = ((zBKD.unsqueeze(2) - zBKD.unsqueeze(1)) ** 2).sum(-1)  # (B, K, K)
                rbf = torch.exp(-d2 / tau)
                R_red = torch.triu(rbf, diagonal=1).sum()
            if quality_weighted_repulsion:
                w = torch.sigmoid(selector.get_logits(z_t, alpha2, prompt_embeds)).view(B, K, 1).detach()
                # Weight pairwise terms by w_i * w_j: high-quality redundant branches repel more.
                R_red = R_red * (w * w.transpose(1, 2)).triu(diagonal=1).sum()  # schematic
        else:
            R_red = z_t.new_zeros(())

        total_loss = sel_loss + repulsion_scale * R_red
        grad = torch.autograd.grad(total_loss, z_t)[0]

        if normalize_guidance_grad:
            grad = grad / (grad.norm(dim=-1, keepdim=True) + 1e-8)
        if guidance_clip is not None:
            grad = grad.clamp(-guidance_clip, guidance_clip)

    # SAME sign as the classifier path: both selector and repulsion contribute to a single loss
    # whose gradient is added to pred_eps with positive scale.
    pred_eps = pred_eps + selector_guidance * sigma2.sqrt() * grad
    pred_x   = predict_start_from_noise(z_t, pred_eps, alpha2)
```

`t_in_window` interprets `start`/`end` as fractions of the denoising trajectory; `sample()` already iterates `time` from `1.0 → 0.0`, so the elapsed fraction is `1 - time[0].item()`. Constant-on-window schedule for now; leave a hook for cosine/linear ramps later.

**Backwards compat:** `selector` defaults to `None`. The new block is fully gated on `selector is not None and selector_guidance != 0.0`. The existing `cls_guidance` branch is untouched.

### 5c. `TransfusionGPT.sample` extensions and explicit return type

Define a single explicit return type used unconditionally — no polymorphic `(2-tuple OR 3-tuple)` shape:

```python
@dataclass
class SampleOutput:
    x_start: Tensor                    # post-processed (unnormalized + L2-normalized) latent, B*K, D)
    x_start_internal: Tensor           # loop-internal normalized latent (B*K, D)  -- selector domain
    generations: List[str]             # FLAT list, length B*K, in (prompt-major, branch-minor) order
    num_branches: int                  # K
    num_prompts: int                   # B
    selector_scores: Optional[Tensor]  # (B, K) or None if selector not used
    selected_branch: Optional[Tensor]  # (B,) long indices into K, set when select_best_plan=True
    pairwise_cosine: Optional[Tensor]  # (B, K, K), set when K>1 and selector is used (diagnostics)
```

Changes to `sample()`:

- Accept `selector`, `prompt_embeds`, `selector_kwargs` (dict packing scale/window/clip/etc.), `num_plan_branches=1`, `return_internal_latents=False` (used by data-collection).
- If K > 1: `input_ids = input_ids.repeat_interleave(K, dim=0)`; `prompt_embeds = prompt_embeds.repeat_interleave(K, dim=0)`. Diffusion is already batched, so the existing loop handles K branches transparently.
- Capture `x_start_internal = x_start.clone().detach()` at the end of the loop, **before** the lines at `transfusion.py:309-311` apply `unnormalize_sentence_emb` + `F.normalize`.
- After the loop, if `selector` is set: score `x_start_internal` via `selector.get_logits(x_start_internal, alpha2≈1, prompt_embeds)` → reshape to `(B, K)` for `selector_scores`. Compute `pairwise_cosine` from `x_start_internal` reshaped to `(B, K, D)`.
- If `select_best_plan`: `selected_branch = selector_scores.argmax(dim=-1)`. Index `x_start`/`x_start_internal`/`input_ids` down from B*K to B before decoding.
- Always return a `SampleOutput`. When `selector` is None, the optional fields are `None` and `num_branches=1`.

### 5d. `interface.py` — explicit return type and B×K grouping

```python
@dataclass
class GenerateResult:
    # Outer list length B (one entry per input prompt). Inner length 1 if select_best_plan
    # OR num_plan_branches==1, else K (branches stay grouped under their parent prompt).
    decoded: List[List[str]]
    selector_scores: Optional[List[List[float]]]   # same outer/inner shape as `decoded`
    selected_branch: Optional[List[int]]            # length B, index into K (or None)
    pairwise_cosine: Optional[List[List[List[float]]]]  # B x K x K (or None)
    plan_stats_path: Optional[str]                  # set when --save_plan_stats was provided
```

- `__init__(selector_path=None)` → loads selector via `load_selector` (mirrors classifier path).
- `generate(...)` accepts and forwards: `selector_guidance`, `num_plan_branches`, `repulsion_scale`, `selector_kwargs`, `select_best_plan`, `save_plan_stats`. It calls `model.sample(...)` once per prompt (current code already loops per-prompt, `interface.py:100-110`) but each call now expands internally to K branches and returns a `SampleOutput`. The interface assembles a `GenerateResult` whose outer index matches input prompt order; inner index matches branch order. If `select_best_plan`, inner length is 1.
- Encodes the prompt once via `self.model.get_sentence_embedding(prompt)` and passes as `prompt_embeds` into `sample()`.
- The flat `generations` list inside `SampleOutput` is reshaped from `[B*K]` to `[B][K]` using `(prompt-major, branch-minor)` order — the same order as `repeat_interleave`. This is documented in the dataclass and tested (§7 unit test 7).

For backwards compatibility with old call sites that expect `List[str]`, the interface also exposes `interface.generate_flat(...) -> List[str]` which flattens the structured result, picking the best-of-K when K>1 and `select_best_plan` is True, or returning all B*K when False. The CLI script `scripts/generate.py` consumes the structured `GenerateResult` and prints accordingly.

---

## 6. Final plan selection & diagnostics

When K > 1 and `--save_plan_stats path.jsonl`, write one JSON line per prompt:

```json
{"prompt_id": "...",
 "prompt": "...",
 "selector_scores": [s1, ..., sK],
 "pairwise_cosine": [[...], ...],
 "selected_branch": k_star,
 "decoded": ["...", ...],
 "correct": null}
```

`correct` is filled in only when a verifier and gold answer are passed (post-hoc analysis tooling — out of scope for this milestone, but the field is reserved). The grouping under each prompt is exactly the inner `[K]` axis of `GenerateResult` (§5d), so a downstream consumer can match scores ↔ decodings unambiguously.

---

## 7. Tests — split unit (CPU, fast, default) vs integration (model checkpoint, opt-in)

### `tests/conftest.py`
- Adds a `--run-integration` pytest option and a `pytest.mark.integration` marker.
- By default, integration tests are **skipped**. They run only when the user passes `--run-integration` (or sets `STARLDM_RUN_INTEGRATION=1`).

### `tests/test_selector_unit.py` — runs on every `pytest` invocation, CPU only, no STAR-LDM checkpoint required

1. **`test_selector_forward_shapes`** — `Selector(...)` accepts `(B, D)` and `(B*K, D)` inputs and returns `(B*K, 1)` logits.
2. **`test_selector_loss_signs`** — `selector.get_score_loss(z, α, p, target=1.0)` decreases when we move `z` along the negative of its z-gradient (sanity check on sign convention).
3. **`test_repulsion_redundancy_loss_decreases_diversity`** — synthesize K=4 latents that are all identical; assert `R_red` is large. Move them apart; assert `R_red` decreases. Confirms the **redundancy** semantics of the loss (high when redundant) and that `autograd.grad(R_red, z)` produces a finite, non-zero gradient with `z.requires_grad=True` (no detach in the path).
4. **`test_repulsion_no_detach`** — call the repulsion subroutine with `z_t.requires_grad_(True)` and confirm `autograd.grad(R_red, z_t)` returns a tensor with non-zero norm (regression guard against accidentally re-introducing `pred_x.detach()` into the rep path).
5. **`test_selector_dataset_loader`** — synthesize a small `.pt` file with the §3 schema, run `train_selector.py` for 2 steps, assert loss is finite and a checkpoint is written.
6. **`test_internal_latent_shape`** — monkeypatch a tiny dummy `TransfusionGPT.sample` (or call the relevant slice directly) to confirm the new `x_start_internal` field has the same shape as `x_start` and is *not* L2-normalized (i.e., its row norms vary, unlike the post-processed `x_start`).
7. **`test_bk_grouping`** — given `num_plan_branches=K=3` and `B=2` prompts, assert `len(GenerateResult.decoded) == 2` and `len(GenerateResult.decoded[i]) == 3`, and that `selector_scores` has matching `[B][K]` layout.

### `tests/test_selector_integration.py` — marked `pytest.mark.integration`, opt-in only

These tests load a real STAR-LDM checkpoint. They require a `STARLDM_TEST_CHECKPOINT` env var pointing to a model dir; otherwise they `pytest.skip`. Reasons: slow, GPU-preferred, large download, not appropriate for a default `pytest` run.

a. **`test_generate_backwards_compat`** — `interface.generate(prompts)` without any selector flags produces identical output (deterministic seed) to a snapshot recorded the first time.
b. **`test_selector_guidance_runs_k1`** and **`test_selector_guidance_runs_k4`** — randomly initialized selector, small `sampling_timesteps=4`, run end-to-end, assert no shape/device errors and finite outputs.
c. **`test_repulsion_runs_k4`** — same with `repulsion_scale > 0`, K=4; assert finite outputs and that the K branches' decoded texts are not all identical (statistical, not strict).
d. **`test_frozen_starldm_params`** — capture `p.grad is None` for all `model.parameters()` after running guided generation; assert all stay `None` (autograd never touched the base model's parameters).

`requirements.txt` already implies pytest is available; if not, the README mentions `pip install pytest`.

---

## 8. Example configs (`configs/`)

Three small YAMLs that mirror existing OmegaConf style:
- `selector_train.yaml` — selector arch + training hyperparams.
- `generate_with_selector.yaml` — single-branch guidance preset.
- `generate_kbranch_repulsion.yaml` — K=8, cosine repulsion, `select_best_plan: true`.

These are read by the scripts via `OmegaConf.load(...)` if `--config` is passed; CLI flags override (same dotlist pattern as `scripts/train.py:35-42`).

---

## 9. README updates

Append a "Selector-guided generation" section with the three example commands from the spec. No new top-level docs files.

---

## Where selector gradients enter the sampler — quick reference

```
TransfusionGPT.sample (transfusion.py:230)
   └── for each (t, t_next):
         └── diffusion_model_predictions(...)         (transfusion.py:420)
              ├── pred_v, pred_x, pred_eps  (existing)
              ├── if cls_guidance != 0: existing classifier path (lines 447-463)
              └── if selector is not None and selector_guidance != 0:   <<< NEW BLOCK
                    ├── sel_loss = BCE(selector(z_t), 1.0)
                    ├── R_red    = sum_{i<j} cos(z_i, z_j)        # redundancy loss on z_t (NOT detached pred_x)
                    │              [or +sum_{i<j} exp(-||z_i-z_j||^2/τ) for L2]
                    ├── grad     = ∇_{z_t} (sel_loss + repulsion_scale * R_red)
                    └── pred_eps += selector_guidance * sqrt(sigma2) * grad     # SAME sign as classifier path
   └── after loop:
        x_start_internal = x_start.clone()       # captured BEFORE post-processing
        # transfusion.py:309-311 post-processing applies; x_start_internal is what the selector sees.
```

---

## Verification (how I'll test end-to-end)

1. `pytest tests/test_selector_unit.py -q` — all unit tests green on CPU, no checkpoint required.
2. `pytest --run-integration tests/test_selector_integration.py -q` (with `STARLDM_TEST_CHECKPOINT` set) — full STAR-LDM generation tests pass.
3. `python -m scripts.generate --model_path checkpoints/star-ldm --prompts "The movie was"` — must produce identical output to pre-change (manual diff with a known seed).
4. Tiny end-to-end loop:
   - Hand-write a 5-prompt JSONL with gold answers.
   - `python -m scripts.collect_selector_data --num_samples_per_prompt 4 --save_noisy_timesteps` → small `.pt` whose `z0_internal` row norms vary (confirms internal-latent capture).
   - `python -m scripts.train_selector --num_steps 50` → `checkpoints/selector/best_model.pt`.
   - `python -m scripts.generate --selector_path checkpoints/selector --selector_guidance 1.0 --num_plan_branches 4 --repulsion_scale 0.1 --select_best_plan --prompts "..."` → runs, prints plan stats grouped per-prompt.
5. `--selector_guidance 0` regression: must equal the no-selector baseline (also a unit test via the integration harness).

---

## Known limitations / TODOs (out of scope for this milestone)

- No learned steerer / RL — supervised selector only.
- Single fixed selector-guidance and repulsion *constant-on-window* schedule. Cosine/linear ramps are a follow-up.
- Verifier registry only ships `exact_match`; GSM8K/MATH/code verifiers later.
- Selector consumes a frozen Sentence-T5 prompt embedding — richer prompt features (GPT-2 hidden states, prompt-conditioned attention) are a follow-up.
- No multi-GPU selector training (single-GPU is fine for the prototype; the data is small).
- Repulsion is computed on `z_t` directly. A `pred_x`-based variant (with a non-detached pred_x reachable via autograd, or a detached-target variant that uses `z_t` only as the differentiation variable) is a research follow-up.

---

## Summary delivered after implementation

- Files changed/added: §"Files to add" + §"Files to modify".
- New CLI flags: §5a.
- How the selector is trained: §4 (BCE + optional pairwise rank, EMA, WSD, modular verifier, on the loop-internal normalized latent).
- Where selector gradients enter the sampler: §"Where selector gradients enter the sampler".
- Limitations / TODOs: §"Known limitations / TODOs".
