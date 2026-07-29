# Experiment plan — a no-SNN gradient-norm clip for Byzantine-robust FL

Date: 2026-07-23. Companion to the activation-clipping handoff. Covers (1) the
offline evidence I gathered on this SSH box (2× A10), (2) the mechanism decision,
(3) the RCP sweeps to run. Scripts: `scripts/experiments/gradnorm_probe.py`,
`gradnorm_analyze.py`, `mechanism_calib.py`. Plots + npz in
`results/activation_clip/gradnorm_probe/`.

---

## 0. The question

The fixed raw-gradient clip **21** was chosen by eyeballing the *max gradient norm
of the SNN*. It works very well (GM, γ=0: 0.16 → 0.77 at f=1) but its justification
is an oracle you only get by training an SNN. **Can we set the same kind of clip
without an SNN and without a norm oracle?** And can a smarter shape (per-layer /
per-neuron / weight-relative) do better?

The handoff already established the hard constraint from the failed adaptive
experiments: **a client-side clip helps only if it is an ABSOLUTE bound, not a
relative statistic** (a running quantile is redundant with ARC and self-references
the scale away). So any candidate must deliver an *absolute* anchor.

---

## 1. Offline evidence (already run here — no RCP budget spent)

`gradnorm_probe.py` reproduces the **real DSGD loop** (10 honest clients share one
global model each round; f=0 plain averaging = no attack; client momentum β=0.9;
lr 0.15, wd 1e-4; the exact `gamma_similarity_niid` partition) and records the RAW
per-client gradient norm — the quantity a raw clip sees, *before* momentum — plus
per-layer norms, ‖W‖ per layer, and per-neuron row-norms, for γ∈{1.0, 0.33, 0.0}.

### 1a. The SNN's "21" is just the honest CNN's own ceiling

Raw **global** grad-norm percentiles (pooled over 500 steps × 10 clients):

| γ | p50 | p80 | p90 | p95 | p99 | max |
|---|----|----|----|----|----|----|
| 1.0  | 1.32 | 1.84 | 2.24 | 2.78 | 3.88 | 6.64 |
| 0.33 | 2.37 | 4.11 | 5.86 | 7.89 | 12.94 | **21.50** |
| 0.0  | 3.32 | 5.99 | 8.61 | 12.31 | 19.99 | 26.67 |

**21 ≈ the honest CNN max at γ=0.33 and ≈ p99 at γ=0.** A 2-minute honest-CNN probe
lands on the same number — **the SNN was never needed.** See
`global_norm_trajectory.png`: honest norms spike early (steps ~30–60, the transient,
reaching the 21 line only under γ=0) then decay to ~1–3. So a fixed cap at ~21:
- clips **< 0.5 %** of honest updates (near no-op on honest training),
- bites **only** during the early non-iid transient — exactly when attacks do the
  most damage — and correctly **stops** clipping late (an adaptive quantile would
  *over*-clip the decay phase; another reason it failed).

**Interpretation: the cap is a stability anchor, not a variance-shaper.** Its job is
to catch the rare early spike that would otherwise blow up the momentum accumulator
(this is also why fixed-21 improves *clean* accuracy). This reframes the value
question: the cap must sit *above* the honest steady state and *at* the honest
ceiling — not lower.

### 1b. Heterogeneity moves the scale — the ceiling is the portable anchor

Raw norm grows ~3× from iid to non-iid (p80 1.8 → 6.0). So an "80th-percentile"
rule is **not** portable across γ (and a client doesn't know its own γ). The
honest **ceiling** (p99…max ≈ 20–27) is far more stable and is what 21 already uses.

### 1c. AGC (weight-relative, NFNets) is the wrong shape here — ruled out

Per-layer ‖g‖/‖W‖ ratios: _c1≈0.04, _f1≈0.10, _c2≈0.15, _f2≈0.22 (γ=0.33) — a **5×
spread**, so a single global λ cannot lightly-clip all layers at once. Simulated on
the measured gradients (`mechanism_calib.py`):

| AGC λ | % honest updates clipped (γ=0) | mean-norm shrink (γ=0) |
|------|------|------|
| 0.05 | 99.7 % | 84 % |
| 0.12 | 96 % | 63 % |
| 0.20 | 88 % | 47 % |

Even λ=0.2 (≫ NFNets' 0.01–0.1) clips most updates and halves the norm — the same
"relative clip touches too much honest mass" pathology that sank the adaptive
quantile. **AGC dropped.** (The one true insight AGC contributes — that all honest
clients share the same ‖W‖ each round, so a weight-relative threshold is common
across them — is real but doesn't rescue the shape mismatch.)

### 1d. Layer-level yes, neuron-level no

Per-neuron (output-unit) grad norms are unimodal with ~1–2 decade spread and **no
outlier-neuron heavy tail** (`per_neuron_hist.png`) → per-neuron caps are not
justified. The dominant structure is the **cross-layer** 5× disparity. Per-layer
honest p99 (γ 0.33+0.0 pooled): **_c1≈1.4, _c2≈4.6, _f1≈12.2, _f2≈11.2**; their
quadrature ≈ 17, consistent with the global ~21 anchor. So the only refinement worth
a run is **per-layer absolute caps**, not per-neuron.

---

## 1e. The calibration ALGORITHM (how the number is chosen, not guessed)

`scripts/experiments/calibrate_clip.py` implements:

> **clip = the smallest absolute cap that leaves ≥ q of honest updates untouched at
> every heterogeneity level.**
> ```
> for γ in γ_grid:
>     run honest DSGD (f=0, all-honest averaging, real γ-partition), steps × seeds
>     record every raw per-client grad norm ‖g‖   (before momentum)
>     P[γ] = q-quantile of pooled ‖g‖
> clip = max over γ of P[γ]
> ```

One knob `q` with a physical meaning ("fraction of honest updates the cap must NOT
touch"); `max over γ` makes it a single portable number. Deterministic, reproducible,
uses the CNN itself — **no SNN, no adversary, no online recalibration.** Output
(q=0.99, seeds 42+43, 500 steps):

| | γ=1.0 | γ=0.33 | γ=0.0 | **cap = maxγ** |
|---|---|---|---|---|
| global | 3.54 | 11.44 | 17.62 | **17.6** |
| _c1 | 0.53 | 0.96 | 1.34 | **1.3** |
| _c2 | 1.34 | 2.98 | 4.39 | **4.4** |
| _f1 | 2.58 | 8.17 | 12.26 | **12.3** |
| _f2 | 2.15 | 7.77 | 12.14 | **12.1** |

**The algorithm outputs ~18, and the SNN's 21 is ~15% above it** — i.e. 21 is a
slightly conservative version of the same honest-ceiling rule. That is the answer to
"how do you choose 21": you don't — you run this and get 17.6, and 21 sits inside its
neighborhood. `q` trades safety margin (higher q / max → larger cap) against stability
margin (lower cap). Per-layer caps use the same rule per layer; quadrature 17.9 ≈ the
global 17.6 (internal consistency check). Numbers written to
`results/activation_clip/gradnorm_probe/clip_calibration.json`, consumed automatically
by the config generator.

## 1f. Why per-layer (box) beats a global cap (ball) in principle

A global cap constrains only the quadrature sum `√(Σ_L‖g_L‖²) ≤ C` — a **ball**. Honest
layer scales differ ~5× (ceilings 1.3 / 4.4 / 12.3 / 12.1), so a global 21 is essentially
*all budget* for f1/f2; a small conv layer can inflate (c1: 1.3→10) and the global norm
barely moves — it passes **unclipped**. A per-layer cap `‖g_L‖ ≤ c_L` is a **box**,
strictly inside the same-radius ball but excluding the anisotropic directions where one
layer dominates. This matters because instability and several attacks concentrate in
specific layers (last-layer excursions under ALIE/label-flip; early-conv under
sign-flip): the box removes them at each layer's native scale; the ball is blind until
they dominate the total. Calibrated identically (`c_L = maxγ P_q(‖g_L‖)`), so it is the
same algorithm, just anisotropic.

## 1g. Self-calibrated warmup clip (2026-07-24 addition — no probe, no SNN, online)

> **Update after measurement (§4b): the prediction below was WRONG, in the opposite
> direction.** W=1 does not collapse at moderate-high heterogeneity — it *wins big*
> there (+18 to +43 pts over W=50 at γ=0.33). Left as-is below to show the reasoning
> trail; see §4b for the actual measured picture and corrected interpretation.

A third calibration route, added after the above: instead of an *externally*
calibrated cap (offline probe or SNN), each client bootstraps its OWN cap from its
own training. For the first `W` steps it only observes its raw grad-norm (no
clipping); once warmup completes, it freezes `cap = max_seen_during_warmup × margin`
and clips every later raw gradient to that fixed value. Same raw position as
`gradient_clip_val`, directly comparable.

**Sanity-checked against the existing probe data before implementing** (reused
`gradnorm_probe.npz`, no new offline run needed):
- The literal reading of "clip to the first gradient" (`W=1`) is **badly
  miscalibrated**: honest raw grad-norm ramps up **4.5×–8×** from step 0 to its peak
  (the heterogeneity transient, ~step 25–60), so a single-sample cap clips **34–95%
  of all later honest steps** — the same over-clipping failure as the adaptive
  quantile / STE.
- A **`W=75`** warmup (with a small `margin=1.1` for seed-variance safety) converges
  to the true honest ceiling with **~0% honest clipping** afterward, across all 3 γ
  tested — i.e. it self-calibrates to essentially the same number the offline
  algorithm finds (§1e), but online, per-client, with no probe or SNN at all.

| config | warmup | margin | expected |
|---|---|---|---|
| `cnn_mnist_selfclip_w75` | 75 | 1.1 | **primary** — should ≈ match gradclip_calib/gradclip21 |
| `cnn_mnist_selfclip_w1` | 1 | 1.0 | **ablation** — should underperform, confirming the warmup matters |

Both full 4-aggregator sweeps. Implemented in `client.py`
(`self_grad_clip_warmup`/`self_grad_clip_margin`, method
`_apply_self_calibrated_clip`), wired through `managers.py`/`train.py`, unit-tested
and smoke-tested through the real pipeline. Launcher: `run_gradclip_selfclip_rcp.sh`
(separate pod from tonight's A/B — lower priority, not required for the mandatory
gradclip21 reproduce).

**Known limitation**: unlike the offline calibrator, which measures on a clean f=0
run, this warmup happens *while attacks are already present* (f is fixed from step
1) — an adversary could in principle try to inflate a client's early gradients to
loosen its cap before switching tactics ("cap poisoning"). None of ALIE/SignFlipping/
IPM specifically target this, so it isn't blocking, but it's a real asymmetry vs the
externally-calibrated caps worth noting for the paper / a future adaptive-adversary
experiment.

## 2. Mechanism decision

| candidate | verdict | why |
|---|---|---|
| adaptive windowed quantile (raw or post-mom) | ✗ (handoff) | relative, redundant w/ ARC, over-clips decay phase |
| AGC weight-relative, per-layer/unit | ✗ (1c) | single λ can't lightly-clip 5×-spread layers; clips ~90 % honest |
| **global fixed cap = offline honest ceiling (~21)** | ✓ **primary** | absolute; = SNN's 21 but no SNN; stability anchor |
| **per-layer fixed caps = each layer's ceiling** | ✓ **refinement** | absolute; respects 5× cross-layer scale |
| per-neuron fixed caps | ✗ (1d) | no outlier neurons; many hyperparams for no signal |
| self-calibrated, W=50 | ✓ variant, no probe/SNN | absolute; ≈ honest ceiling; best at LOW heterogeneity (§4b) |
| **self-calibrated, W=1 ("first gradient")** | ✓✓ **best measured so far at γ=0.33** | tight absolute cap; +18 to +43pts over W=50 at γ=0.33 (§4b, GM-only pilot — needs 4-agg confirmation) |

**Headline for the paper:** the clip is an *absolute cap at the honest gradient
ceiling*, and that ceiling is a **cheap, model-intrinsic, offline-measurable
property of the CNN** — no SNN, no online oracle, no per-γ tuning. That closes the
"where does 21 come from?" gap and keeps the honest-side-shaping thesis intact.

---

## 3. Code implemented (this session — needs commit + image rebuild)

- `client.py`: new `layer_grad_clip_val` = `{module_name: cap}`; caps each named
  module's raw grad-norm to its own absolute ceiling, same position as the global
  `gradient_clip_val`. Unit-tested (clips to cap; no-op when unset).
- `managers.py` `get_honest_clients_layer_grad_clip_val`; wired in `train.py`.
- `generate_activation_clip_configs.py`: `make_config(..., aggregators=)` override +
  four new configs (below). Regenerate with the usual command.

Generated configs (`configs/activation_clip/`):

All five use the SAME sweep as the rest of the activation_clip family (4
aggregators GM/CC/TrMean/MultiKrum, NNM→ARC, 3 attacks, 5 seeds, 500 steps, lr
0.15) so best_test (max over aggregators) is directly comparable.

Every clip value except 21 is an ALGORITHM output read from
`clip_calibration.json` (21 is hardcoded only because it is the SNN value being
reproduced).

| config | clip value | source |
|---|---|---|
| `cnn_mnist_gradclip21` | 21 (global) | SNN value (reproduce baseline) |
| `cnn_mnist_gradclip_calib` | ~17.6 (global) | algorithm, q0.99 |
| `cnn_mnist_gradclip_qlow` | ~9 (global) | algorithm, q0.90 (tight bracket) |
| `cnn_mnist_gradclip_qhigh` | ~27 (global) | algorithm, q1.0/max (loose bracket) |
| `cnn_mnist_layerclip` | per-layer {~1.3,4.4,12.3,12.1} | algorithm, q0.99 per layer |

All five: 4 aggregators (GM/CC/TrMean/MultiKrum), NNM→ARC, 3 attacks, 5 seeds, 500
steps, lr 0.15 — identical to the rest of the activation_clip family.

Per-layer caps + `gradclip_calib` are generated from
`clip_calibration.json`; regenerate that with `calibrate_clip.py` then re-run the
config generator to propagate any change of `q`.

All share the standard sweep: f∈{0..5}, γ∈{1,0.66,0.33,0}, NNM→ARC pre-agg,
attacks {Optimal ALIE neg1, SignFlipping, Optimal IPM}, cnn_mnist, 500 steps.

---

## 4. RCP run plan (2 balanced pods, 4 GPUs each)

**Prereq (non-negotiable): rebuild + push the Docker image** — client/managers/train
changed and code is baked in (`COPY .`); only `results/` persists on the PVC.

All 5 configs (gradclip21, gradclip_calib, gradclip_qlow, gradclip_qhigh, layerclip)
run the SAME full sweep. Because each RCP pod is capped at 4 GPUs, we balance by
**splitting the heterogeneity grid** instead of the config list, so two 4-GPU pods
finish at ~the same time:

- **POD A** (`run_gradclip_A_rcp.sh`): all 5 configs on γ∈{1.0, 0.66}
- **POD B** (`run_gradclip_B_rcp.sh`): all 5 configs on γ∈{0.33, 0.0}

Each pod = 5 mechanisms × 2 γ = exactly half the work. Both write into the SAME
`results_directory` per config (per-setting folders are uniquely named by γ, so the
halves merge); pods use `--no_plots`. After BOTH finish, regenerate all heatmaps once
(read-only) with `run_gradclip_plots.sh`.

Submit (4 GPUs/pod, v100):
```
runai submit --name gradclip-a -i registry.rcp.epfl.ch/byzfl-snn/byzfl_snn:latest \
  -g 4 --node-pools v100 --pvc dcl-scratch:/home/bendouro/results \
  -- bash run_gradclip_A_rcp.sh 80
runai submit --name gradclip-b -i registry.rcp.epfl.ch/byzfl-snn/byzfl_snn:latest \
  -g 4 --node-pools v100 --pvc dcl-scratch:/home/bendouro/results \
  -- bash run_gradclip_B_rcp.sh 80
```

### What each result decides
- gradclip21 across all 4 aggregators (vs the prior GM-only run) → anchor is
  aggregator-robust; the mandatory reproduce.
- layerclip vs gradclip21: does per-layer capping beat a scalar? Predicted **≈ tie**
  (honest barely clipped by either), but layerclip may edge ahead in the γ=0 transient.
- qlow / calib / qhigh (7.7 / 17.6 / 27) vs 21: the accuracy-vs-cap curve; a
  plateau/peak near the honest ceiling validates the calibration rule (the real
  deliverable — *how to pick the number*).

---

## 4b. MEASURED results — self-calibrated clip pilots (SSH, 2026-07-24)

First measured results for the self-calibrated warmup clip (§1g), run on the SSH box
(2× A10), **GeometricMedian only, 3 seeds** (coarse — the γ=0 collapse cells are
near-bimodal, so treat those as directional). `best_test` = worst-case over the 3
attacks.

**w1** (`cnn_mnist_selfclip_w1_gm3`, warmup=1, margin=1.0 — "clip to the literal first
gradient"):

| γ \ f | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 1.0  | 0.97 | 0.97 | 0.96 | 0.95 | 0.95 | 0.94 |
| 0.66 | 0.98 | 0.95 | 0.92 | 0.89 | 0.87 | 0.88 |
| 0.33 | 0.99 | 0.96 | 0.93 | 0.90 | 0.83 | 0.55 |
| 0.0  | 0.98 | 0.81 | 0.24 | 0.21 | 0.16 | 0.16 |

**w50** (`cnn_mnist_selfclip_w50_gm3`, warmup=50, margin=1.0):

| γ \ f | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 1.0  | 0.99 | 0.99 | 0.98 | 0.98 | 0.98 | 0.97 |
| 0.66 | 0.99 | 0.98 | 0.97 | 0.94 | 0.85 | 0.90 |
| 0.33 | 0.99 | 0.78 | 0.50 | 0.54 | 0.43 | 0.18 |
| 0.0  | 0.99 | 0.16 | 0.19 | 0.16 | 0.16 | 0.16 |

Reference — old fixed grad-clip-21 (GM): γ=0.33 f1–5 = 0.89/0.89/0.74/0.65/0.68;
γ=0.0 f1–3 = 0.77/0.45/0.40.

**Finding — the prediction in §1g was WRONG, and in the opposite direction than
expected.** I predicted w1's short warmup (tight cap) would collapse at high
heterogeneity from under-clipped signal, and w50 (cap ≈ honest ceiling) would recover
it. The measured w1-vs-w50 diff shows the reverse pattern:

| region | w1 vs w50 | interpretation |
|---|---|---|
| γ=1.0 / 0.66 (low het.) | w50 slightly ahead (+2 to +5 pts) | expected: less clipping costs nothing when variance reduction isn't needed |
| **γ=0.33 (moderate-high het.)** | **w1 dramatically ahead (+18 to +43 pts, every f)** | tighter absolute cap = much stronger honest-variance reduction = the dominant robustness lever here, not signal loss |
| γ=0.0, f=1 | w1 far ahead (0.81 vs 0.16) | same effect, one step before the wall |
| γ=0.0, f≥2 | both ≈ 0.16–0.24 (tied, at floor) | the known γ=0/f≥3 wall — no clip choice crosses it (§6) |

So **more aggressive absolute clipping is a stronger robustness lever than "calibrate to
the honest ceiling"** across most of the grid, and only costs a small amount in the
easy (near-iid) regime where it isn't needed anyway. This is a genuinely different
conclusion from the earlier "cap = honest ceiling" framing built around fixed-21/
gradclip_calib/layerclip — those calibrate to barely touch honest gradients (<0.5%
clipped); w1 clips 34–95% of honest steps and wins anyway at γ=0.33. The two families
(externally-calibrated-to-the-ceiling vs. aggressively self-clipped) may be doing
different things and are worth reporting as a genuine tension, not resolved in favor
of one story.

**Caveats before trusting this**: GM-only, 3 seeds (γ=0.33 cells especially — 0.18,
0.43, 0.50 are in the noisy transition zone). Not yet known whether this holds for
CC/TrMean/MultiKrum — priority next step is the full 4-aggregator sweep
(`cnn_mnist_selfclip_w1.json`, already generated, 5 seeds) before treating this as
more than a strong lead. If it holds, natural follow-up: is w1 near-optimal, or would
an even tighter self-calibrated cap (or the same warmup=1 idea with a smaller margin)
do better still?

---

## 5. Not doing (with reasons, so they aren't re-litigated)
- No τ sweep / no AGC / no per-neuron caps / no adaptive-quantile completion —
  ruled out in §1–2.
- No new attempt at the γ=0, f≥3 wall (~0.16 for everyone; no mechanism crosses it).
- CIFAR ramp-vs-hardtanh (from the handoff) is a *separate* activation experiment,
  orthogonal to this clip study.
