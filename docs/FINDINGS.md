# What's wrong with the current model — diagnostic write-up

Audit of the `entropy_specificity_onGreaterThan1` pipeline, 2026-07-27. Baseline
under review: `Models/best_model_w2048_b5_f64_mask.pt` (job 7192179), reported
test Pearson 0.680 / Spearman 0.533 / R² 0.444, pred-vs-observed slope 0.49.

Everything below is measured, not assumed; the commands to reproduce are at the
bottom. The short version: **two of the four data-prep steps are corrupting the
dataset, and the model is not clearly beating a five-number linear baseline.**

---

## 1. The headline: the CNN barely beats sequence composition

A linear model on 5 composition numbers (GC fraction, CpG fraction, CpG
observed/expected, and two squares) over the central 256 bp:

| model | params | test Pearson | test Spearman |
|---|---|---|---|
| GC/CpG linear baseline | 6 | **0.604** | **0.458** |
| 3-mer ridge (central 256 bp) | 65 | 0.597 | 0.455 |
| `best_model_w2048_b5_f64_mask` CNN | ~14 M | 0.680 | 0.533 |

A 14M-parameter conv net trained on 4.4M examples buys **+0.08 Pearson over six
numbers**. Worse, when the CNN's own predictions are regressed on those same
composition features, **R² = 0.59** — most of what the network computes is a GC/CpG
detector with extra steps.

Nothing in the repo previously measured this, so there was no way to tell whether
architecture changes were helping. `src/baselines.py` now computes all of it.

## 2. The model doesn't reach the trivial spatial ceiling either

For each test region, take the mean label of its neighbours within ±h bp,
**excluding itself**. That uses no sequence at all — it is purely the label's
spatial autocorrelation — and it is an upper bound on any model whose effective
resolution is h:

| neighbourhood | Pearson | Spearman |
|---|---|---|
| ±256 bp | 0.886 | 0.835 |
| ±1024 bp (= the current 2048 bp window) | **0.785** | 0.751 |

The CNN scores 0.680. **It does not reach the ceiling implied by its own window
width**, so the gap is the model's, not the label's. There is real headroom here;
the project is not up against an information limit.

This also reframes the range compression that earlier work chased. If a model can
only resolve the local average, the expected pred-vs-true slope is
`var(smoothed)/var(y)` = **0.63** at ±1024 bp. The observed slope is 0.49. So the
compression is mostly a *resolution and capacity* symptom, not primarily the
score=1.0 pile-up — which is consistent with spike rebalancing and inverse-density
weighting having failed to fix it (see `git log`, both reverted).

## 3. The label is a ~256 bp local property — 2048 bp is too much context

Composition fit vs. how much sequence is used:

| context | Pearson | | context | Pearson |
|---|---|---|---|---|
| 64 bp | 0.558 | | 512 bp | 0.602 |
| 128 bp | 0.590 | | 1024 bp | 0.588 |
| **256 bp** | **0.605** | | 2048 bp | 0.556 |

It peaks at 256 bp and **decays monotonically** past it. Adding wide-context
features on top of the central 256 bp is worth +0.005. The 2048 bp window is ~8×
more sequence than the signal supports, and it is not free:

- the labelled 16 bp becomes 1 of 64 positions after 5 pool-2 blocks, and
  `AttentionPool` returns a softmax-weighted **average** over all of them;
- 8× more sequence to memorize — these runs overfit by **epoch 2** (`logs/train_model.7433705.out`:
  train loss falls monotonically while val loss goes 0.36 → 1.31);
- 8× the compute per epoch (~45 min/epoch).

Direct check on the trained model: shifting the input window by 16 bp — one whole
region — moves the prediction by 0.021 on a prediction std of 0.194
(r = 0.978 between shifted and unshifted). It is close to blind at the scale the
label actually varies on.

## 4. Data bug: ~38% of all rows are duplicates

The source bedgraph is *annotated*, not deduplicated. It emits one row per
(interval × overlapping annotation), so an interval covered by two genes appears
twice with the same score:

```
chr1  944320  944336  1  3primeUTR  SAMD11
chr1  944320  944336  1  3primeUTR  NOC2L     <- same interval, same score
```

Measured on the current splits:

| split | rows | unique intervals | duplicated |
|---|---|---|---|
| train | 4,410,272 | 2,704,771 | **38.7%** |
| val | 789,121 | 493,430 | 37.5% |
| test | 371,894 | 237,573 | 36.1% |

Up to 9 copies of a single locus. Zero groups disagree on the score, so they are
genuinely redundant. Why it matters:

- **Multiplicity is not random.** It tracks annotation density, which
  anti-correlates with the label: multiplicity 1 → mean score 0.79, 3 → 0.63,
  5 → 0.49. Training therefore oversamples the low-score tail 3–5×, silently, in
  the opposite direction from the `BALANCE_SPIKE`/`LOSS_WEIGHTING` levers that
  were added to correct the distribution.
- **Val/test metrics are per-row, not per-locus** — every reported number weights
  a locus by how many genes overlap it. That is not the quantity being reported.

Fixed by `DEDUP_REGIONS` (default on) in `src/prepare_data.py`.

## 5. Data bug: the MACS2 filter discarded ~48% of the data on garbage peaks

`data/peaks.bed` contains **46 peaks**. They look like this:

```
chr2   667440    203739568    span=203.1 Mb
chr4   384672    158171456    span=157.8 Mb
chr1   944144    112524464    span=111.6 Mb
chr1   21818816  31506240     span=9.7 Mb     <- nested inside the one above
```

Multi-megabase, overlapping, nested. These are not peaks; they are arbitrary
chromosome blocks, and filtering on them dropped **10.67M regions → 5.57M**.
Coverage is wildly uneven across chromosomes (chr16 kept 2.2 Mb of 90 Mb; chr2
kept a single 203 Mb "peak"), so the splits are arbitrary genomic blocks.

**Root cause:** the bedgraph is sorted by **annotation class first**, not by
position. It walks the whole genome once per class —

```
line        1  3primeUTR  chr1:944144
line   126517  5primeUTR  chr1:924208     <- genome restarts
line   329053  CA         chr1:29344      <- and again, 12 times total
```

— giving 5.77M chromosome switches and 143k backward position steps.
`macs2 bdgpeakcall` consumes a coverage track as a position-sorted stream, so it
was fed nonsense. **It exited 0 and wrote a plausible-looking narrowPeak file
anyway**, which is why this went unnoticed.

Fixed three ways in `src/prepare_data.py`: the filter is now opt-in
(`PEAK_FILTER`, default **off**), the bedgraph is position-sorted before MACS2 is
called, and `check_peaks_sane()` warns when a peak file has the degenerate shape.

Note that a 0–1 bounded score is arguably not what `bdgpeakcall` is for at all —
sorting is necessary but may not be sufficient. Inspect `data/peaks.bed` before
re-enabling.

## 6. Smaller things

- **The LR scheduler watched the wrong metric.** `ReduceLROnPlateau` always
  stepped on `val_loss` even when `--monitor pearson` selected the checkpoint. In
  the raw-label runs val loss exploded while Pearson held flat, so the LR was
  repeatedly halved because of a metric nobody was optimizing. Fixed in
  `src/trainer.py`.
- **The annotation columns were thrown away.** Feature class alone spans mean
  score 0.38 (5primeUTR) / 0.41 (PLS) to 0.97 (CA) — a class-only predictor is a
  real baseline, and per-class breakdowns are the natural way to see *where* the
  model fails. Now carried through to the parquets for stratified eval (never as
  a model input — the model still predicts from sequence alone).
- **`TARGET_CLIP = (0.02, 0.98)` caps the achievable slope.** With 41–45% of rows
  at exactly 1.0 and a sigmoid that can never exceed 0.98, some compression is
  built in by construction. Minor next to §2–3, but it means slope ≈ 1 is not
  reachable on this head.
- **`filter_by_macs2_peaks` was O(rows × peaks)** — a Python row-loop over 10.7M
  rows. Now a vectorized per-chromosome binary search.

---

## What to try, in expected-value order

1. **Regenerate the data** with `DEDUP_REGIONS=True, PEAK_FILTER=False`. This
   roughly doubles usable regions (5.57M → ~10.7M pre-dedup, ~6.6M unique) and
   removes the silent reweighting. Everything downstream changes, so do it first —
   and re-baseline, because metrics before and after are not comparable.
2. **Sweep context width**: `--crop 256`, `512`, `1024` against the 2048 baseline.
   No data regeneration needed. §3 predicts 256–512 bp wins on both accuracy and
   ~8× compute; that also buys back the epochs currently lost to overfitting.
3. **`--center-pool`**, so the head reads the features over the labelled region
   instead of only the window-wide average (§3).
4. **Then** revisit the loss. A rank/correlation loss and a censored head for the
   1.0 pile-up are still worth testing — but §2 says the model is not yet at the
   ceiling its *inputs* allow, so loss engineering is premature.
5. Always report `python src/baselines.py` alongside. A change that does not move
   the model away from the composition baseline has not learned sequence grammar.

## Reproducing

```bash
python src/baselines.py --sample 20000                 # §1, §2, §3
python src/baselines.py --which smoothed --halves 64 256 1024
awk -F'\t' '{f[$5]++; s[$5]+=$4} END{for(k in f) print k, f[k], s[k]/f[k]}' \
    data/entropy_specificity_onGreaterThan1_stitched_annotated_complete.bedgraph
tail -n +2 data/peaks.bed | awk -F'\t' '{print $1,$2,$3,($3-$2)/1e6" Mb"}'
```
