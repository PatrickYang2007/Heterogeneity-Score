"""Per-base sequence attributions for a trained HeterogeneityScoreModel.

Answers "which bases in this window drove the predicted score, and by how much"
with one number per position (and, optionally, one per position PER BASE, which
is what sequence logos and TF-MoDISco consume).

Three methods, deliberately kept side by side because on THIS architecture they
are not interchangeable:

  ism        In-silico mutagenesis. Substitute every base at every position and
             re-run the model. Exact by construction -- no backprop rules, no
             reference-propagation assumptions, nothing to be wrong about. Costs
             3*L forward passes per window, which for this small conv stack is
             seconds on a GPU. This is the GROUND TRUTH the others are checked
             against.
  deeplift   Captum DeepLIFT (Rescale rule) against dinucleotide-shuffled
             references. One backward pass per reference -- orders of magnitude
             cheaper than ISM, so it is the only option at whole-genome scale.
             But see "Why DeepLIFT is approximate here" below: this model has an
             attention pool, and DeepLIFT has no rule for it.
  gradxinput Plain gradient * input. One backward pass, no reference at all, so
             it cannot separate "this base matters" from "this window is
             GC-rich". A control -- though on the cropped model it beat DeepLIFT
             against ISM, which is itself evidence about the attention pool.

Two things about this model specifically break the naive recipe, and both are
silent failures -- you get plausible-looking numbers either way:

1. ATTRIBUTE THE LOGIT, NOT THE BOUNDED OUTPUT.
   The per-region head ends in a sigmoid and ~41% of the labels sit at exactly
   1.0, so a fit model pushes many windows toward the upper rail. Measured on
   3000 val windows with best_model_w2048_b5_f64_mask_monpearson: sigmoid'(z)
   has median 0.079 against a maximum of 0.25, 10th percentile 0.028, minimum
   0.008 -- a 30x spread across windows.

   Be precise about what that does and does not break. The sigmoid is a monotone
   scalar map applied once at the very end, so it multiplies EVERY position in a
   window by the SAME constant. Within one window the attribution profile is
   therefore unchanged in shape (correlation with the logit version is exactly 1;
   see tests). What it wrecks is comparability ACROSS windows: two windows whose
   bases matter equally can differ 30x in apparent attribution magnitude purely
   from where they sit on the sigmoid, and the shrinkage is systematic, not
   random -- the label==1.0 windows are squashed ~1.6x harder than the median.
   So anything that pools attributions across windows (ranking the strongest
   positions genome-wide, feeding TF-MoDISco, averaging profiles) is distorted,
   while a single-window logo is not.

   Attributing the pre-sigmoid logit removes the squashing and changes no
   ranking. `LogitView` below does it, and it is the default for every method;
   --keep-sigmoid opts out and warns.
   For a raw-label model (bounded=False) the head is already linear: no-op.

2. USE A DINUCLEOTIDE-SHUFFLED REFERENCE, NOT ZEROS.
   DeepLIFT attributions are always "relative to a reference". An all-zero
   one-hot is not a sequence at all, and attributing against it mostly recovers
   "this window has bases in it", which for this task means it recovers GC
   content. That is a real problem here and not a hypothetical one: a 6-parameter
   GC/CpG linear model already reaches test Pearson ~0.60 against the CNN's
   ~0.69, so composition alone explains most of the signal. Shuffling each window
   while PRESERVING its dinucleotide (hence mono-nucleotide, hence GC and CpG)
   content makes the attributions answer the question actually worth asking:
   what does the model use BEYOND composition? `dinuc_shuffle()` implements the
   Altschul-Erikson doublet-preserving shuffle.

WHY DEEPLIFT IS APPROXIMATE HERE (read before trusting it)
   DeepLIFT's Rescale rule decomposes linear ops and elementwise nonlinearities.
   AttentionPool is neither: it computes `(x * softmax(g(x))).sum(dim=1)`, a
   MULTIPLICATION of two branches that both depend on x. DeepLIFT has no rule
   for that, so Captum just lets raw gradients through it, and the summation-to-
   delta (completeness) guarantee no longer holds. Captum will not warn you.

   Separately, Captum 0.7.0's SUPPORTED_NON_LINEAR table has no entry for
   nn.GELU -- and its `_can_register_hook` SILENTLY SKIPS unlisted modules rather
   than raising. Since every conv block in this model is BatchNorm -> GELU ->
   Conv, an unpatched Captum degrades to plain gradients at every activation
   while still calling itself DeepLIFT. `register_gelu_rule()` installs the
   generic Rescale rule for GELU and is called automatically before attributing.
   It is worth real accuracy, not just tidiness: patched vs unpatched DeepLIFT
   correlate only r=0.86 with each other, and agreement with exact ISM goes
   0.625 -> 0.714.

   Measured agreement with exact ISM, on two checkpoints:

                                     512 bp crop      full 2048 bp
                                     (12 windows)     (256 windows)
       DeepLIFT (patched)                0.71             0.68
       gradient * input                  0.77             0.56
       DeepLIFT (as captum ships it)     0.63              --
       completeness median rel. err.     0.14             0.20
       completeness p90  rel. err.       0.71             1.29

   Read two things off that. First, a correct DeepLIFT satisfies completeness to
   numerical precision, so a 14-20% median error IS the attention pool's missing
   rule, showing up directly; a p90 above 1.0 means those windows' attributions
   sum further from delta than zero would. Second, THE RANKING OF THE TWO
   APPROXIMATIONS FLIPS between models -- neither can be trusted on reputation,
   and neither clears r ~ 0.7 against the exact answer.

   ISM has no such caveat and is cheap: 256 windows x 2048 bp, all three
   methods, ran in 3 min on one GPU. Prefer it. The others are here to be
   checked against it, which is why the default is `--method ism --method
   deeplift` rather than deeplift alone.

   Re-run that comparison after any architecture change -- especially one that
   removes or replaces AttentionPool, which is the thing breaking DeepLIFT here.

Example:
  python src/attribute.py data/val_w2048.parquet \
      --weights Models/best_model_w2048_b5_f64_mask_monpearson.pt \
      --window 2048 --num-blocks 5 --num-filters 64 \
      --method ism --method deeplift --n-regions 64 --out attributions.npz
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from model import load_model, region_mask_channel, center_crop
from config import (REGION_WIDTH as CFG_REGION_WIDTH,
                    pool_for_window, region_mask_enabled, in_channels_for,
                    check_flags_match_checkpoint)

BASES = np.array(list("ACGT"))


def _device_of(model):
    """Device the model's tensors live on, or CPU for a parameter-less module.

    `next(model.parameters())` raises StopIteration on a module with no
    parameters -- and inside a generator (pytest's hooks, for one) that surfaces
    as an unrelated RuntimeError rather than anything diagnosable.
    """
    for p in model.parameters():
        return p.device
    for b in model.buffers():
        return b.device
    return torch.device("cpu")


# ------------------------------- references --------------------------------

def dinuc_shuffle(tokens, num_shufs, rng):
    """Altschul-Erikson doublet-preserving shuffle of an integer token array.

    Returns an (num_shufs, L) int array. Every output has EXACTLY the same
    dinucleotide counts as the input (and therefore the same base composition,
    GC content and CpG count), but the arrangement is randomized. That makes it
    the right null for this model: attributions against it measure what the
    network extracts from base ORDER, not from composition, which a 6-parameter
    GC/CpG regression already captures most of.

    Implementation is the standard Eulerian-path construction: treat each
    adjacent pair as a directed edge between base vertices, shuffle each vertex's
    outgoing edge list while PINNING its last edge, then walk the graph. Pinning
    the last edge is what guarantees the walk is a valid Eulerian path and does
    not strand itself; without it this silently produces truncated sequences.
    """
    tokens = np.asarray(tokens)
    chars, idx = np.unique(tokens, return_inverse=True)

    # For each vertex, the positions of the successors of its occurrences.
    next_inds = []
    for t in range(len(chars)):
        mask = idx[:-1] == t                    # last token has no successor
        next_inds.append(np.flatnonzero(mask) + 1)

    out = np.empty((num_shufs, len(tokens)), dtype=tokens.dtype)
    for s in range(num_shufs):
        shuffled = []
        for t in range(len(chars)):
            order = np.arange(len(next_inds[t]))
            if len(order) > 1:
                order[:-1] = rng.permutation(len(order) - 1)   # pin the last edge
            shuffled.append(next_inds[t][order])

        counters = [0] * len(chars)
        result = np.empty_like(idx)
        pos = 0
        result[0] = idx[0]
        for j in range(1, len(idx)):
            t = idx[pos]
            pos = shuffled[t][counters[t]]
            counters[t] += 1
            result[j] = idx[pos]
        out[s] = chars[result]
    return out


def shuffled_references(onehot, num_refs, rng):
    """Build `num_refs` dinucleotide-shuffled one-hot references for one window.

    `onehot` is (4, L) float. Positions that are all-zero (encoded N) are left
    all-zero in every reference so they contribute no delta and pick up no
    attribution.
    """
    seq = onehot.argmax(axis=0)
    is_n = onehot.sum(axis=0) == 0
    # Shuffle only the real bases; splice the N positions back in afterwards so
    # they stay N and the doublet statistics are computed over actual sequence.
    real = seq[~is_n]
    if len(real) < 2:
        return np.repeat(onehot[None], num_refs, axis=0)

    shuf = dinuc_shuffle(real, num_refs, rng)
    refs = np.zeros((num_refs, 4, len(seq)), dtype=onehot.dtype)
    cols = np.flatnonzero(~is_n)
    for r in range(num_refs):
        refs[r, shuf[r], cols] = 1.0
    return refs


# ------------------------------ model plumbing -----------------------------

class LogitView(nn.Module):
    """Wrap the model so it returns the PRE-sigmoid logit.

    See point 1 in the module docstring: with ~41% of labels at 1.0 a fit model
    sits near the sigmoid's upper rail, where the derivative vanishes and every
    gradient-based attribution collapses to ~0 on precisely the windows worth
    explaining. Monotone, so it changes no ranking. A no-op for a linear
    (bounded=False) head.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.bounded = model.bounded

    def forward(self, x):
        if not self.bounded:
            return self.model(x)
        # Re-run the body without the final sigmoid rather than inverting it:
        # logit(sigmoid(z)) round-trips to inf once sigmoid saturates in float32.
        m = self.model
        for block in m._blocks:
            x = block(x)
        if m.final_norm is not None:
            x = m.final_norm(x)
        pooled = m.pool(x)
        if m.center_pool:
            pooled = torch.cat([pooled, x[:, :, x.shape[-1] // 2]], dim=1)
        return m.fc(pooled)


def register_gelu_rule():
    """Teach Captum's DeepLIFT the Rescale rule for nn.GELU.

    Captum 0.7.0 ships rules for ReLU/ELU/LeakyReLU/Sigmoid/Tanh/Softplus/
    MaxPool/Softmax only, and `DeepLift._can_register_hook` returns False for
    anything else WITHOUT warning -- so on this model (every block is
    BatchNorm -> GELU -> Conv) unpatched Captum quietly falls back to plain
    gradients at every activation and still reports the result as DeepLIFT.

    Captum's own `nonlinear` helper is already generic -- it is the Rescale rule
    `grad_out * (out - out_ref) / (in - in_ref)` with a gradient fallback when
    the denominator underflows -- so GELU needs no special math, only an entry in
    the table. Idempotent.
    """
    from captum.attr._core import deep_lift as _dl
    _dl.SUPPORTED_NON_LINEAR.setdefault(nn.GELU, _dl.nonlinear)
    return nn.GELU in _dl.SUPPORTED_NON_LINEAR


def build_inputs(df, crop, region_mask, region_width):
    """Encode a frame of sequences to (N, C, L) float32, cropped and masked
    exactly the way GenomicDataset does during training."""
    seq = df["sequence"]
    if crop and crop < len(seq.iat[0]):
        lo = len(seq.iat[0]) // 2 - crop // 2
        seq = seq.str.slice(lo, lo + crop)
    seqs = seq.str.upper().tolist()

    L = len(seqs[0])
    arr = np.frombuffer("".join(seqs).encode(), dtype=np.uint8).reshape(len(seqs), L)
    onehot = np.zeros((len(seqs), 4, L), dtype=np.float32)
    for i, b in enumerate(b"ACGT"):
        onehot[:, i, :] = (arr == b)
    x = torch.from_numpy(onehot)
    if region_mask:
        mask = region_mask_channel(L, region_width).expand(len(seqs), 1, -1)
        x = torch.cat([x, mask], dim=1)
    return x


# -------------------------------- methods ----------------------------------

@torch.no_grad()
def ism_attributions(model, x, batch_size=256):
    """Exact in-silico mutagenesis for one window.

    `x` is (C, L). Returns (4, L) HYPOTHETICAL attributions: entry [b, i] is
    f(sequence with position i set to base b) minus the mean over the four bases
    at that position. Mean-centering per position is what makes ISM comparable to
    DeepLIFT/SHAP hypothetical contributions, which are also centered by
    construction; it also removes the (large, uninformative) constant f(wildtype)
    offset so the numbers reflect local preference rather than window identity.

    Only the four base channels are mutated -- the region-mask channel is
    positional metadata, not sequence, and perturbing it would measure something
    that has no genomic meaning.
    """
    device = _device_of(model)
    C, L = x.shape
    variants = torch.empty((4 * L, C, L), dtype=x.dtype)
    for b in range(4):
        block = x.unsqueeze(0).repeat(L, 1, 1)
        pos = torch.arange(L)
        block[pos, :4, pos] = 0.0
        block[pos, b, pos] = 1.0
        variants[b * L:(b + 1) * L] = block

    preds = torch.empty(4 * L)
    for i in range(0, len(variants), batch_size):
        chunk = variants[i:i + batch_size].to(device)
        preds[i:i + batch_size] = model(chunk).squeeze(-1).float().cpu()

    hyp = preds.view(4, L).numpy()
    return hyp - hyp.mean(axis=0, keepdims=True)


def deeplift_attributions(model, x, refs, batch_size=32):
    """Captum DeepLIFT for one window, averaged over dinucleotide-shuffled refs.

    Returns (hyp, observed, delta_out) where
      hyp        (4, L) hypothetical contributions, mean-centered per position
      observed   (L,)   contribution of the base actually present
      delta_out  float  f(x) - mean_r f(ref_r), the quantity attributions should
                        sum to if completeness held

    Rather than take Captum's attributions directly we ask for the raw
    MULTIPLIERS (via custom_attribution_func) and combine them here. That buys
    the hypothetical contributions: for a counterfactual base b at position i the
    attribution is `mult[b,i] - sum_c mult[c,i] * ref[c,i]`, which needs the
    multipliers and the reference separately. Multiplying back by (x - ref) and
    summing over channels recovers the ordinary observed-base attribution.
    """
    from captum.attr import DeepLift

    # Idempotent, and done here rather than only in run() so that calling this
    # function directly cannot silently get the gradient-fallback behavior.
    register_gelu_rule()

    device = _device_of(model)
    C, L = x.shape
    n = len(refs)

    ref_t = torch.zeros((n, C, L), dtype=x.dtype)
    ref_t[:, :4] = torch.from_numpy(refs)
    if C > 4:
        # The region mask is identical in input and reference, so its delta is 0
        # and it takes no attribution -- which is correct, it is not sequence.
        ref_t[:, 4:] = x[4:].unsqueeze(0)

    inp = x.unsqueeze(0).expand(n, -1, -1).contiguous()

    dl = DeepLift(model)
    mults = []
    for i in range(0, n, batch_size):
        a, b = inp[i:i + batch_size].to(device), ref_t[i:i + batch_size].to(device)
        with warnings.catch_warnings():
            # Captum warns that the model's forward is called with hooks attached;
            # expected, and noisy once per window.
            warnings.simplefilter("ignore")
            m = dl.attribute(a, baselines=b,
                             custom_attribution_func=lambda mult, i_, b_: mult)
        mults.append(m.detach().cpu())
    mult = torch.cat(mults).numpy()[:, :4]              # (n, 4, L)

    refs4 = ref_t[:, :4].numpy()
    x4 = x[:4].numpy()

    # Observed: standard DeepLIFT attribution, summed over channels -> per base.
    observed = (mult * (x4[None] - refs4)).sum(axis=1).mean(axis=0)

    # Hypothetical: what a counterfactual base at each position would have got.
    base_term = (mult * refs4).sum(axis=1, keepdims=True)   # (n, 1, L)
    hyp = (mult - base_term).mean(axis=0)                   # (4, L)
    hyp = hyp - hyp.mean(axis=0, keepdims=True)

    with torch.no_grad():
        fx = model(x.unsqueeze(0).to(device)).squeeze().item()
        fr = np.concatenate([
            model(ref_t[i:i + batch_size].to(device)).squeeze(-1).cpu().numpy()
            for i in range(0, n, batch_size)]).mean()
    return hyp, observed, float(fx - fr)


def gradxinput_attributions(model, x):
    """gradient * input on the (logit) output. Returns (hyp, observed).

    Included as a control, not a recommendation: it is a first-order local
    approximation, so it is the method most damaged by any saturation left in
    the network, and unlike DeepLIFT it has no reference at all -- which means it
    cannot separate "this base matters" from "this window is GC-rich".
    """
    device = _device_of(model)
    inp = x.unsqueeze(0).to(device).requires_grad_(True)
    out = model(inp).squeeze()
    grad = torch.autograd.grad(out, inp)[0][0, :4].cpu().numpy()
    hyp = grad - grad.mean(axis=0, keepdims=True)
    observed = (grad * x[:4].numpy()).sum(axis=0)
    return hyp, observed


def project(hyp, onehot):
    """Collapse (4, L) hypothetical contributions to (L,) by reading off the
    base that is actually present. N positions (all-zero column) give 0."""
    return (hyp * onehot).sum(axis=0)


# ------------------------------- driver ------------------------------------

def run(weights, input_parquet, out_path, methods, n_regions, num_refs,
        window, num_filters, num_blocks, ker_size, dropout, bounded,
        pool, region_mask, region_width, crop, center_pool, seed,
        keep_sigmoid, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    check_flags_match_checkpoint(weights, crop=crop, center_pool=center_pool)

    base = load_model(weights, device, in_channels=in_channels_for(region_mask),
                      num_filters=num_filters, num_blocks=num_blocks,
                      ker_size=ker_size, dropout=dropout, pool=pool,
                      bounded=bounded, center_pool=center_pool)
    model = base if keep_sigmoid else LogitView(base).to(device).eval()
    if keep_sigmoid and bounded:
        print("WARNING: attributing the saturating sigmoid output. Gradient-based "
              "methods will read ~0 on high-scoring windows; see module docstring.")

    df = pd.read_parquet(input_parquet,
                         columns=["chrom", "start", "end", "sequence", "score"])
    if n_regions and n_regions < len(df):
        df = df.sample(n=n_regions, random_state=seed).sort_index()
    print(f"attributing {len(df):,} regions from {os.path.basename(input_parquet)}")

    x_all = build_inputs(df, crop, region_mask, region_width)
    L = x_all.shape[-1]
    onehot_all = x_all[:, :4].numpy()
    rng = np.random.default_rng(seed)

    if "deeplift" in methods:
        assert register_gelu_rule(), "failed to register the GELU rescale rule"

    results = {m: np.zeros((len(df), 4, L), dtype=np.float32) for m in methods}
    deltas, sums = [], []

    for i in range(len(df)):
        x = x_all[i]
        if "ism" in methods:
            results["ism"][i] = ism_attributions(model, x, batch_size=batch_size)
        if "deeplift" in methods:
            refs = shuffled_references(onehot_all[i], num_refs, rng)
            hyp, obs, delta = deeplift_attributions(model, x, refs)
            results["deeplift"][i] = hyp
            deltas.append(delta)
            sums.append(obs.sum())
        if "gradxinput" in methods:
            results["gradxinput"][i] = gradxinput_attributions(model, x)[0]
        if (i + 1) % 25 == 0 or i + 1 == len(df):
            print(f"  {i + 1}/{len(df)}")

    report = {}

    # Completeness: DeepLIFT's summation-to-delta property. It CANNOT hold
    # exactly here (AttentionPool multiplies two x-dependent branches and has no
    # DeepLIFT rule), so this is not a pass/fail assertion -- it is the
    # measurement of how badly that missing rule bites on this model.
    if "deeplift" in methods:
        deltas, sums = np.array(deltas), np.array(sums)
        denom = np.maximum(np.abs(deltas), 1e-8)
        rel = np.abs(sums - deltas) / denom
        report["completeness_median_rel_err"] = float(np.median(rel))
        report["completeness_p90_rel_err"] = float(np.percentile(rel, 90))
        print(f"\ncompleteness |sum(attr) - delta| / |delta|: "
              f"median {np.median(rel):.3f}, p90 {np.percentile(rel, 90):.3f}")

    # ISM is exact, so any other method is only as good as its agreement with it.
    if "ism" in methods:
        for m in methods:
            if m == "ism":
                continue
            per = [np.corrcoef(project(results["ism"][i], onehot_all[i]),
                               project(results[m][i], onehot_all[i]))[0, 1]
                   for i in range(len(df))]
            per = np.array(per)
            per = per[~np.isnan(per)]
            report[f"ism_vs_{m}_pearson_median"] = float(np.median(per))
            print(f"per-base agreement with ISM ({m}): median r = "
                  f"{np.median(per):.3f}  (IQR {np.percentile(per, 25):.3f}"
                  f"-{np.percentile(per, 75):.3f})")

    payload = {f"hyp_{m}": results[m] for m in methods}
    payload.update({f"attr_{m}": np.stack([project(results[m][i], onehot_all[i])
                                           for i in range(len(df))])
                    for m in methods})
    payload["onehot"] = onehot_all.astype(np.int8)
    payload["chrom"] = df["chrom"].to_numpy().astype("U8")
    payload["start"] = df["start"].to_numpy()
    payload["end"] = df["end"].to_numpy()
    payload["score"] = df["score"].to_numpy()
    np.savez_compressed(out_path, **payload)
    print(f"\nwrote {out_path}  ({len(df)} regions x {L} bp, methods: {', '.join(methods)})")
    return report


def main():
    p = argparse.ArgumentParser(
        description="Per-base attributions (ISM / DeepLIFT / grad*input) for a "
                    "trained HeterogeneityScoreModel")
    p.add_argument("input_parquet")
    p.add_argument("--weights", default="best_model.pt")
    p.add_argument("--out", default="attributions.npz")
    p.add_argument("--window", type=int, required=True,
                   help="window the model was trained on (sets pooling)")
    p.add_argument("--method", action="append", dest="methods",
                   choices=["ism", "deeplift", "gradxinput"],
                   help="repeatable; default: ism + deeplift, so the "
                        "approximation can be checked against the exact answer")
    p.add_argument("--n-regions", type=int, default=64,
                   help="random subsample of rows to attribute (0 = all). ISM is "
                        "3*L forward passes per row, so keep this small at first")
    p.add_argument("--num-refs", type=int, default=20,
                   help="dinucleotide-shuffled references per window (DeepLIFT)")
    p.add_argument("--num-filters", type=int, default=32)
    p.add_argument("--num-blocks", type=int, default=3)
    p.add_argument("--ker-size", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--raw-label", dest="raw_label", action="store_true")
    p.add_argument("--crop", type=int, default=None)
    p.add_argument("--center-pool", dest="center_pool", action="store_true")
    p.add_argument("--keep-sigmoid", action="store_true",
                   help="attribute the squashed output instead of the logit. "
                        "Not recommended -- see the module docstring")
    args = p.parse_args()

    methods = args.methods or ["ism", "deeplift"]
    run(weights=args.weights,
        input_parquet=args.input_parquet,
        out_path=args.out,
        methods=methods,
        n_regions=args.n_regions,
        num_refs=args.num_refs,
        window=args.window,
        num_filters=args.num_filters,
        num_blocks=args.num_blocks,
        ker_size=args.ker_size,
        dropout=args.dropout,
        bounded=not (args.aggregate or args.raw_label),
        pool=pool_for_window(args.window),
        region_mask=region_mask_enabled(args.aggregate),
        region_width=CFG_REGION_WIDTH,
        crop=args.crop,
        center_pool=args.center_pool and not args.aggregate,
        seed=args.seed,
        keep_sigmoid=args.keep_sigmoid,
        batch_size=args.batch_size)


if __name__ == "__main__":
    main()
