import math

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr


def inverse_density_weights(y, num_bins=50, clip=(0.1, 10.0), smooth=1.0):
    """Build a score->weight lookup that upweights rare score values.

    The per-region labels are heavily imbalanced: ~41% pile up at 1.0 while the
    low-score tail is sparse. Under plain MSE every row contributes equally, so
    the dense high-score region dominates the gradient and the model regresses
    toward the mean (compressed output range, best-fit slope well below 1).

    Weighting each row by the inverse frequency of its score bin makes the rare
    low-score examples contribute as much total gradient as the common ones,
    countering the compression WITHOUT discarding data the way spike
    downsampling does. `smooth` adds a pseudocount so empty/near-empty bins
    don't blow up; `clip` bounds the weight ratio; weights are normalized to
    mean 1 so the overall loss scale (and a comparable LR) is preserved.

    Returns (bin_edges, bin_weights) for use with torch.bucketize.
    """
    y = np.asarray(y, dtype=np.float64)
    lo, hi = float(y.min()), float(y.max())
    edges = np.linspace(lo, hi, num_bins + 1)
    counts, _ = np.histogram(y, bins=edges)
    density = counts + smooth
    w = 1.0 / density
    w = w / w.mean()                      # provisional; renormalize by mass below
    idx = np.clip(np.digitize(y, edges[1:-1]), 0, num_bins - 1)
    w = np.clip(w, clip[0], clip[1])
    # Normalize so the mean weight over the actual data is 1.0 (keeps loss scale).
    w = w / w[idx].mean()
    return edges, w


class Trainer:
    def __init__(self, model, train_loader, val_loader, num_epochs, lr=1e-3,
                 weight_decay=1e-4, grad_clip=1.0, patience=10, early_stopping=True,
                 checkpoint_path="best_model.pt", label_clip=None,
                 loss_weighting=None, monitor="loss"):
        self.model = model
        self.checkpoint_path = checkpoint_path
        # Optional (lo, hi) range the regression target is squashed into before
        # the loss. With a sigmoid head and labels that pile up at exactly 1.0,
        # an unclipped target forces the output onto the saturating rail; e.g.
        # (0.02, 0.98) keeps the optimum reachable on the steep part of the
        # curve. None disables clipping (used by the unbounded linear head).
        self.label_clip = label_clip
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr,
                                            weight_decay=weight_decay)
        # The scheduler must plateau on the SAME metric that selects the
        # checkpoint. It used to always watch val_loss (mode='min') even when
        # monitor="pearson", and the two can diverge sharply: in the raw-label
        # runs val_loss blew up 0.36 -> 1.31 while val Pearson held ~0.5, so the
        # LR kept getting halved because of a metric nobody was optimizing.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max' if monitor == 'pearson' else 'min',
            factor=0.5, patience=5)
        self.loss_fn = nn.MSELoss()
        self.grad_clip = grad_clip
        self.patience = patience
        self.early_stopping = early_stopping
        self.best_val_loss = float('inf')
        self.best_val_corr = float('-inf')
        # Which validation metric drives checkpointing / early stopping.
        #   "loss"    -> save the minimum-MSE model (original behavior). Under a
        #               spiked label this favors the more mean-regressed model.
        #   "pearson" -> save the maximum-Pearson model, i.e. the one that best
        #               preserves rank/spread -- the metric actually reported.
        assert monitor in ("loss", "pearson")
        self.monitor = monitor
        # Optional inverse-density loss weighting (see inverse_density_weights).
        # Built from the TRAIN labels; applied per batch by looking each target
        # up in the bin->weight table. None -> plain (unweighted) MSE.
        self.loss_weighting = loss_weighting
        self._bin_edges = None
        self._bin_weights = None
        if loss_weighting == "inv_density":
            edges, weights = inverse_density_weights(train_loader.dataset.y.numpy())
            # bucketize uses the interior edges; a target in bin i gets weights[i].
            self._bin_edges = torch.tensor(edges[1:-1], dtype=torch.float32,
                                           device=self.device)
            self._bin_weights = torch.tensor(weights, dtype=torch.float32,
                                              device=self.device)

    def _clip_target(self, y):
        """Squash the regression target into label_clip, or pass it through when
        clipping is disabled (unbounded linear head)."""
        return y.clamp(*self.label_clip) if self.label_clip is not None else y

    def _weighted_mse(self, preds, target, y_true):
        """MSE, optionally weighted by inverse label density (keyed on the true,
        unclipped label so the sparse tail is upweighted)."""
        if self._bin_weights is None:
            return self.loss_fn(preds, target)
        idx = torch.bucketize(y_true, self._bin_edges)
        w = self._bin_weights[idx]
        return (w * (preds - target) ** 2).mean()

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            target = self._clip_target(y)
            self.optimizer.zero_grad()
            preds = self.model(x).squeeze(1)
            # Weighting keys on the true (unclipped) label; target may be clipped.
            loss = self._weighted_mse(preds, target, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def val_epoch(self):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                preds = self.model(x).squeeze(1)
                target = self._clip_target(y)
                total_loss += self.loss_fn(preds, target).item()
                all_preds.extend(preds.cpu().tolist())
                # Pearson is reported against the true (unclipped) labels so it
                # reflects real performance, not agreement with the squashed target.
                all_targets.extend(y.cpu().tolist())

        avg_loss = total_loss / len(self.val_loader)
        # A collapsed model outputs one constant value, so the predictions have
        # zero variance and Pearson is undefined (pearsonr returns nan with a
        # ConstantInputWarning). Detect that explicitly and report nan + a
        # pred_std of 0.0 so the collapse is visible in the log instead of
        # crashing the "best" tracking downstream.
        pred_std = float(np.std(all_preds))
        if pred_std == 0.0 or np.std(all_targets) == 0.0:
            corr = float('nan')
        else:
            corr, _ = pearsonr(all_preds, all_targets)
        return avg_loss, corr, pred_std

    def fit(self):
        train_losses = []
        val_losses = []
        epochs_no_improve = 0

        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch()
            val_loss, corr, pred_std = self.val_epoch()
            # Step the scheduler on whatever metric is being monitored. A
            # collapsed epoch gives corr=nan; feed the worst possible value
            # instead so it counts as "no improvement" rather than poisoning the
            # scheduler's best-so-far comparison.
            if self.monitor == "pearson":
                self.scheduler.step(-1.0 if math.isnan(corr) else corr)
            else:
                self.scheduler.step(val_loss)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            lr = self.optimizer.param_groups[0]['lr']
            print(f"epoch {epoch+1}/{self.num_epochs}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  pearson={corr:.4f}  pred_std={pred_std:.4f}  "
                  f"lr={lr:.2e}")

            # Decide whether this epoch is the new best under the monitored
            # metric. For "pearson" we maximize (and skip nan collapse epochs);
            # for "loss" we minimize val_loss (always finite). Whichever metric
            # is monitored also drives early stopping, so the saved checkpoint
            # and the stopping criterion always agree.
            improved = False
            if self.monitor == "pearson":
                if not math.isnan(corr) and corr > self.best_val_corr:
                    self.best_val_corr = corr
                    improved = True
            else:  # "loss"
                if not math.isnan(corr) and corr > self.best_val_corr:
                    self.best_val_corr = corr        # still tracked for the report
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    improved = True

            if improved:
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                epochs_no_improve += 1
                if self.early_stopping and epochs_no_improve >= self.patience:
                    print(f"early stopping at epoch {epoch+1} "
                          f"(no {self.monitor} improvement for {self.patience} epochs)")
                    break

        return train_losses, val_losses
