import argparse
import torch
import pandas as pd

from model import load_model, region_mask_channel, center_crop
from prepare_data import one_hot_encode
from config import (REGION_WIDTH as CFG_REGION_WIDTH,
                    pool_for_window, region_mask_enabled, in_channels_for,
                    check_flags_match_checkpoint)


def predict(weight_file, input_parquet, output_file,
            num_filters=32, num_blocks=3, ker_size=5, dropout=0.3, batch_size=64,
            bounded=True, pool=2, region_mask=False, region_width=16,
            crop=None, center_pool=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --crop changes the input width but not any weight shape, so a mismatch
    # would silently score the wrong context instead of raising. Warn on it.
    check_flags_match_checkpoint(weight_file, crop=crop, center_pool=center_pool)

    # bounded must match training: pass --aggregate for summed-bin model weights.
    # num_filters/num_blocks must match the trained model's width/depth, and pool
    # must match too (pool=2 for windowed models, pool=1 for the raw 16 bp path);
    # a mismatch changes the layer shapes and load_state_dict will fail. The
    # region-mask channel must match too: a masked model takes in_channels=5.
    model = load_model(weight_file, device, in_channels=in_channels_for(region_mask),
                       num_filters=num_filters, num_blocks=num_blocks,
                       ker_size=ker_size, dropout=dropout, pool=pool, bounded=bounded,
                       center_pool=center_pool)

    df = pd.read_parquet(input_parquet)
    sequences = df["sequence"].str.upper().tolist()
    encoded = one_hot_encode(sequences)
    x = torch.from_numpy(encoded.transpose(0, 2, 1).astype("float32"))
    # Same center crop the dataset applies during training; the mask below is then
    # built at the cropped length so it still marks the central region.
    x = center_crop(x, crop)
    if region_mask:
        # Append the same central-region marker the dataset adds during training.
        mask = region_mask_channel(x.shape[-1], region_width).expand(x.shape[0], 1, -1)
        x = torch.cat([x, mask], dim=1)

    preds = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = x[i:i + batch_size].to(device)
            preds.extend(model(batch).squeeze(1).cpu().tolist())

    out = df[["chrom", "start", "end"]].copy()
    if "score" in df.columns:
        out["true_score"] = df["score"].values
    out["predicted_score"] = preds
    out.to_csv(output_file, sep="\t", index=False)
    print(f"wrote {len(out):,} predictions to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained HeterogeneityScoreModel")
    parser.add_argument("input_parquet", help="parquet file with sequence (and optionally score) columns")
    parser.add_argument("--weights", default="best_model.pt", help="model weights file (default: best_model.pt)")
    parser.add_argument("--output", default="predictions.tsv", help="output TSV file (default: predictions.tsv)")
    parser.add_argument("--window", type=int, required=True,
                        help="window/bin size the model was trained on; sets pooling "
                             "(pool=2 for any window, pool=1 for the raw 16 bp path)")
    parser.add_argument("--num-filters", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=3,
                        help="must match the trained model's depth")
    parser.add_argument("--ker-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--aggregate", action="store_true",
                        help="weights are from a summed-bin model (linear output, no sigmoid)")
    parser.add_argument("--raw-label", dest="raw_label", action="store_true",
                        help="weights are from a raw-signal model (per-region, linear "
                             "output); keeps the region-mask channel")
    # Must match training (see train.py --crop / --center-pool).
    parser.add_argument("--crop", type=int, default=None,
                        help="feed only the central N bp (must match training)")
    parser.add_argument("--center-pool", dest="center_pool", action="store_true",
                        help="weights were trained with the center-anchored head")
    args = parser.parse_args()

    # Match train.py / eval_report.py: any window pools by 2, the raw 16 bp path
    # (window unset / 0) uses no pooling. The per-region path carries the
    # region-mask channel; the summed-bin path does not. Must match how the
    # weights were trained.
    pool = pool_for_window(args.window)
    # The raw-signal model is per-region (keeps the mask channel) but linear-headed,
    # so bounded is off for both the summed-bin and raw paths.
    region_mask = region_mask_enabled(args.aggregate)

    predict(
        weight_file=args.weights,
        input_parquet=args.input_parquet,
        output_file=args.output,
        num_filters=args.num_filters,
        num_blocks=args.num_blocks,
        ker_size=args.ker_size,
        dropout=args.dropout,
        batch_size=args.batch_size,
        bounded=not (args.aggregate or args.raw_label),
        pool=pool,
        region_mask=region_mask,
        region_width=CFG_REGION_WIDTH,
        crop=args.crop,
        center_pool=args.center_pool and not args.aggregate,
    )


if __name__ == "__main__":
    main()
