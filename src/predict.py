import argparse
import torch
import numpy as np
import pandas as pd

from model import HeterogeneityScoreModel
from prepare_data import one_hot_encode


def predict(weight_file, input_parquet, output_file,
            num_filters=32, num_blocks=3, ker_size=5, dropout=0.3, batch_size=64,
            bounded=True, pool=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # bounded must match training: pass --aggregate for summed-bin model weights.
    # num_filters/num_blocks must match the trained model's width/depth, and pool
    # must match too (pool=2 for windowed models, pool=1 for the raw 16 bp path);
    # a mismatch changes the layer shapes and load_state_dict will fail.
    model = HeterogeneityScoreModel(dropout=dropout, ker_size=ker_size,
                                  num_filters=num_filters, num_blocks=num_blocks,
                                  pool=pool, bounded=bounded)
    model.load_state_dict(torch.load(weight_file, map_location=device))
    model = model.to(device)
    model.eval()

    df = pd.read_parquet(input_parquet)
    sequences = df["sequence"].str.upper().tolist()
    encoded = one_hot_encode(sequences)
    x = torch.from_numpy(encoded.transpose(0, 2, 1).astype("float32"))

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
    args = parser.parse_args()

    # Match train.py / eval_report.py: any window pools by 2, the raw 16 bp path
    # (window unset / 0) uses no pooling.
    pool = 2 if args.window else 1

    predict(
        weight_file=args.weights,
        input_parquet=args.input_parquet,
        output_file=args.output,
        num_filters=args.num_filters,
        num_blocks=args.num_blocks,
        ker_size=args.ker_size,
        dropout=args.dropout,
        batch_size=args.batch_size,
        bounded=not args.aggregate,
        pool=pool,
    )


if __name__ == "__main__":
    main()
