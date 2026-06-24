import torch
import torch.nn as nn
from scipy.stats import pearsonr


class Trainer:
    def __init__(self, model, train_loader, val_loader, num_epochs, lr=1e-3,
                 weight_decay=1e-4, grad_clip=1.0, patience=10, early_stopping=True,
                 checkpoint_path="best_model.pt"):
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr,
                                            weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5)
        self.loss_fn = nn.MSELoss()
        self.grad_clip = grad_clip
        self.patience = patience
        self.early_stopping = early_stopping
        self.best_val_loss = float('inf')
        self.best_val_corr = float('-inf')

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(x).squeeze(1)
            loss = self.loss_fn(preds, y)
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
                total_loss += self.loss_fn(preds, y).item()
                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(y.cpu().tolist())

        avg_loss = total_loss / len(self.val_loader)
        corr, _ = pearsonr(all_preds, all_targets)
        return avg_loss, corr

    def fit(self):
        train_losses = []
        val_losses = []
        epochs_no_improve = 0

        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch()
            val_loss, corr = self.val_epoch()
            self.scheduler.step(val_loss)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            lr = self.optimizer.param_groups[0]['lr']
            print(f"epoch {epoch+1}/{self.num_epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  pearson={corr:.4f}  lr={lr:.2e}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

            if corr > self.best_val_corr:
                self.best_val_corr = corr
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                epochs_no_improve += 1
                if self.early_stopping and epochs_no_improve >= self.patience:
                    print(f"early stopping at epoch {epoch+1} "
                          f"(no val improvement for {self.patience} epochs)")
                    break

        return train_losses, val_losses
