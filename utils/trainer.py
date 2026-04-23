import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -----------------------------------------------------------------------------------------
# --------------------------------------- Early Stopping ----------------------------------
# -----------------------------------------------------------------------------------------
class EarlyStopper():
    """
    Stops training when validation loss stops improving.

    Args:
        patience (int): Epochs to wait before stopping. Default 15.
        min_delta (float): Minimum improvement to reset counter. Default 0.001.
        save_path (str): Path to save the best model weights.
    """
    def __init__(self, patience=15, min_delta=0.001, save_path='best_model.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.save_path = save_path

    def early_stop(self, validation_loss, model):
        """Returns True if training should stop, saves model if improved."""
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
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
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr)
        self.history = {'train_loss': [], 'val_loss': []}

    def fit(self, X_train, y_train, X_val, y_val,
            num_epochs=1000, patience=15, save_path='best_model.pt'):
        """
        Trains the model with early stopping on validation loss.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
            num_epochs (int): Maximum epochs. Default 1000.
            patience (int): Early stopping patience. Default 15.
            save_path (str): Path to save the best model weights.

        Returns:
            dict: Training history with 'train_loss' and 'val_loss' lists.
        """
        n_neg = float((y_train == 0).sum())
        n_pos = float((y_train == 1).sum())
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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
            with torch.no_grad():
                for X, y in val_loader:
                    X = X.to(DEVICE)
                    y = y.to(DEVICE)
                    y_pred = self.model(X)
                    loss = criterion(y_pred.view(-1), y.view(-1))
                    val_loss += loss.item()

            # ---------- EPOCH SUMMARY ----------
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            t = time.time()
            print(f"Epoch {epoch+1}/{num_epochs} Complete - Time: {t-t_0:.2f} seconds")
            print(f"Epoch {epoch+1}/{num_epochs} - Train: {train_loss:.4f} - Val: {val_loss:.4f}")
            print("___________________________________")

            if es.early_stop(val_loss, self.model):
                print(f"Early stopping at epoch: {epoch+1}")
                break

        T = time.time()
        elapsed_time = T - T_0
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        print(f"Total elapsed time: {minutes} Minutes and {seconds:.2f} seconds")
        print("___________________________________")
        self.model.load_state_dict(torch.load(save_path, map_location=DEVICE))
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
        """Plots train and validation loss curves."""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.history['train_loss'], label='Train Loss')
        ax.plot(self.history['val_loss'], label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{self.model.__class__.__name__} — Learning Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
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