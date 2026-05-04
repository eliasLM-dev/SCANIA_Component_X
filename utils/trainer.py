import time
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

from torch.utils.data import WeightedRandomSampler

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.85, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, 
            reduction='none'
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)
        return (focal_weight * bce).mean()


# -----------------------------------------------------------------------------------------
# --------------------------------------- Early Stopping ----------------------------------
# -----------------------------------------------------------------------------------------
class EarlyStopper():
    """
    Stops training when validation AUC-PR stops improving.

    Args:
        patience (int): Epochs to wait before stopping. Default 15.
        min_delta (float): Minimum improvement to reset counter. Default 0.001.
        save_path (str): Path to save the best model weights.
    Returns:
        bool: True if training should stop, False otherwise.
    """
    def __init__(self, patience=15, min_delta=0.001, save_path='best_model.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_auc_pr = 0.0
        self.save_path = save_path

    def early_stop(self, auc_pr, model):
        """Returns True if training should stop, saves model if improved.
        Args:
            auc_pr (float): Current epoch's validation AUC-PR.
            model (nn.Module): The model to save.
        Returns:
            bool: True if training should stop, False otherwise.
        """
        if auc_pr > self.best_auc_pr + self.min_delta:
            self.best_auc_pr = auc_pr
            self.counter = 0
            # torch.save(model.state_dict(), self.save_path)
            return False

        self.counter += 1
        return self.counter >= self.patience



# -----------------------------------------------------------------------------------------
# --------------------------------------- Base Trainer ------------------------------------
# -----------------------------------------------------------------------------------------
class BaseTrainer():
    """
    Trainer for binary classification models in PyTorch.

    Handles training loop, early stopping, evaluation, and prediction
    for any nn.Module that outputs a single logit.

    Args:
        model (nn.Module): The model to train.
        lr (float): Learning rate. Default 0.001.
        batch_size (int): Batch size. Default 32.
        clip_grad (float | None): Gradient clipping norm. None disables it.
        optimizer: Custom optimizer. Defaults to Adam if None.
    """
    def __init__(self, model, lr=0.001, batch_size=32, clip_grad=None, optimizer=None):
        self.model = model.to(DEVICE)
        self.lr = lr
        self.batch_size = batch_size
        self.clip_grad = clip_grad
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        self.history = {'train_loss': [], 'val_loss': [], 'val_auc_pr': []}

    def fit(self, X_train, y_train, X_val, y_val,
            num_epochs=1000, patience=15, save_path='best_model.pt'):
        """
        Trains the model with early stopping on validation AUC-PR.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
            num_epochs (int): Maximum epochs. Default 1000.
            patience (int): Early stopping patience. Default 15.
            save_path (str): Path to save the best model weights.

        Returns:
            dict: Training history with 'train_loss', 'val_loss', 'val_auc_pr' lists.
        """
        criterion = FocalLoss(alpha=0.85, gamma=2.0)
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
            batch_size=self.batch_size, shuffle=True
        )

        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
            batch_size=self.batch_size, shuffle=False
        )

        es = EarlyStopper(patience=patience, min_delta=0.001, save_path=save_path)
        T_0 = time.time()

        for epoch in range(num_epochs):
            t_0 = time.time()

            # ---------- TRAINING ----------
            self.model.train()
            train_loss = 0

            for X, y in train_loader:
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                self.optimizer.zero_grad()
                y_pred = self.model(X)
                loss = criterion(y_pred.view(-1), y.view(-1))
                loss.backward()
                if self.clip_grad:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.optimizer.step()
                train_loss += loss.item()

            # ---------- VALIDATION ----------
            self.model.eval()
            val_loss = 0.0
            val_probs_all = []
            val_labels_all = []
            with torch.no_grad():
                for X, y in val_loader:
                    X = X.to(DEVICE)
                    y = y.to(DEVICE)
                    y_pred = self.model(X)
                    loss = criterion(y_pred.view(-1), y.view(-1))
                    val_loss += loss.item()
                    val_probs_all.append(torch.sigmoid(y_pred.view(-1)))
                    val_labels_all.append(y)

            val_loss /= len(val_loader)
            val_probs = torch.cat(val_probs_all).cpu().numpy()
            val_labels = torch.cat(val_labels_all).cpu().numpy()
            val_auc_pr = average_precision_score(val_labels, val_probs)

            # ---------- EPOCH SUMMARY ----------
            train_loss /= len(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_auc_pr'].append(val_auc_pr)

            t = time.time()
            print(f"Epoch {epoch+1}/{num_epochs} - Time: {t-t_0:.2f}s - Train: {train_loss:.4f} - Val: {val_loss:.4f} - Val AUC-PR: {val_auc_pr:.4f}")
            print("___________________________________")

            if es.early_stop(val_auc_pr, self.model):
                print(f"Early stopping at epoch: {epoch+1}")
                break

        T = time.time()
        elapsed_time = T - T_0
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        print(f"Total elapsed time: {minutes} Minutes and {seconds:.2f} seconds")
        print("___________________________________")
        #self.model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        return self.history

    def predict(self, X, threshold=0.5):
        """Returns (predictions, probabilities) for input X at given threshold."""
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(torch.FloatTensor(X).to(DEVICE))
            probabilities = torch.sigmoid(y_pred).squeeze().cpu().numpy()
            predictions = (probabilities > threshold).astype(int)
        return predictions, probabilities

    def evaluate(self, X, y_true, threshold=0.5):
        """Prints and returns (metrics dict, predictions, probabilities) at given threshold."""
        predictions, probabilities = self.predict(X, threshold)
        metrics = {
            'Recall': recall_score(y_true, predictions),
            'Precision': precision_score(y_true, predictions),
            'F1': f1_score(y_true, predictions),
            'AUC-ROC': roc_auc_score(y_true, probabilities),
            'AUC-PR': average_precision_score(y_true, probabilities),
        }
        print("___________________________________________")
        print('Standard Metrics:')
        for k, v in metrics.items():
            print(f"{k:12s}: {v:.4f}")
        return metrics, predictions, probabilities

    def plot_history(self):
        """Plots train loss, val loss, and val AUC-PR curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{self.model.__class__.__name__} — Loss Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(self.history['val_auc_pr'], label='Val AUC-PR', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('AUC-PR')
        ax2.set_title(f'{self.model.__class__.__name__} — Validation AUC-PR')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_cm(self, X, y_true, threshold=0.5):
        """Plots confusion matrix at given threshold."""
        predictions, _ = self.predict(X, threshold)
        cm = confusion_matrix(y_true, predictions)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        classes = ['No Repair', 'Repair']
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               title=f'Confusion Matrix\n(Threshold: {threshold})',
               ylabel='Actual Status',
               xlabel='Predicted Status')

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.show()

    def save(self, vehicle_ids, X, model_path, predictions_path, threshold=0.5):
        """Saves model weights to model_path and predictions CSV to predictions_path."""
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        predictions, probabilities = self.predict(X, threshold)
        pd.DataFrame({
            'vehicle_id': list(vehicle_ids),
            'probability': probabilities,
            'binary_prediction': predictions
        }).to_csv(predictions_path, index=False)
        print(f"Predictions saved to {predictions_path}")




# -----------------------------------------------------------------------------------------
# -------------------------- Random Search ------------------------------------------------
# -----------------------------------------------------------------------------------------


def random_search(model_class, param_grid, model_kwargs_fn, 
                  X_train, y_train, X_val, y_val,
                  n_iter=10, num_epochs=1000, patience=15,
                  save_dir='models', model_name='model', clip_grad = None,
                    seed=42):
    """
    Random search over hyperparameters for a BaseTrainer-compatible model.
    
    Args:
        model_class: The model class to instantiate (e.g. LSTMModel, TCNModel)
        param_grid: Dict of hyperparameter names to lists of values
        model_kwargs_fn: Function that takes params dict and returns model kwargs
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_iter: Number of random trials
        num_epochs: Max epochs per trial
        patience: Early stopping patience
        save_dir: Directory to save trial weights
        model_name: Name prefix for saved files
        seed: Random seed
    
    Returns:
        best_params: Dict of best hyperparameters found
        best_auc_pr: Best AUC-PR achieved
        results: DataFrame of all trials
    """
    random.seed(seed)
    best_auc_pr = 0.0
    best_params = None
    results = []

    for i in range(n_iter):
        params = {k: random.choice(v) for k, v in param_grid.items()}
        print(f"\nTrial {i+1}/{n_iter}: {params}")

        model = model_class(**model_kwargs_fn(params))
        trainer = BaseTrainer(model, lr=params['lr'], clip_grad=clip_grad)
        trainer.fit(
            X_train, y_train, X_val, y_val,
            num_epochs=num_epochs,
            patience=patience,
            save_path=Path(save_dir) / f'{model_name}_trial_{i}.pt'
        )

        _, probs = trainer.predict(X_val)
        auc_pr = average_precision_score(y_val, probs)
        print(f"Trial {i+1} AUC-PR: {auc_pr:.4f}")

        results.append({**params, 'trial': i+1, 'auc_pr': auc_pr})
        pd.DataFrame(results).to_csv(Path(save_dir) / f'{model_name}_search_results.csv', index=False)

        if auc_pr > best_auc_pr:
            best_auc_pr = auc_pr
            best_params = params
            print(f"New best {model_name}: {params} — AUC-PR: {auc_pr:.4f}")

    print(f"\nBest {model_name} params: {best_params}")
    print(f"Best {model_name} AUC-PR: {best_auc_pr:.4f}")

    return best_params, best_auc_pr, pd.DataFrame(results)