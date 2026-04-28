import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
import torch
from torch.utils.data import Dataset

# -----------------------------------------------------------------------------------------
# --------------------------------------- Cumulative Features -----------------------------
# -----------------------------------------------------------------------------------------

def get_cumulative_cols(df, cols, threshold=0.95):
    df = df.sort_values(['vehicle_id', 'time_step'])
    cumulative = []
    for col in cols:
        if df[col].dtype != 'object':
            diffs = df.groupby('vehicle_id')[col].diff().dropna()
            active_diffs = diffs[diffs != 0]

            if len(active_diffs) > 0 and (active_diffs > 0).mean() > threshold:
                cumulative.append(col)

    print(f"Analyzing cumulative features with threshold {threshold}...")            
    print(f"Cumulative features: {cumulative}")
    print(f"Non-cumulative features: {[col for col in cols if col not in cumulative]}")
                
    return cumulative



# -----------------------------------------------------------------------------------------
# --------------------------------------- LR Data -----------------------------------------
# -----------------------------------------------------------------------------------------

def prepare_lr_data(df, label_col):
    """
    Prepares input data for Logistic Regression by extracting the last timestep per vehicle.

    Args:
        df (pd.DataFrame): Raw sequential vehicle data.
        label_col (str): Name of the label column.

    Returns:
        tuple: (X, y) where X is a DataFrame of features and y is a Series of labels.
    """
    X = df.groupby('vehicle_id').last()
    y = X[label_col]
    X.drop(columns=[label_col, 'time_step'], inplace=True)
    return X, y


# -----------------------------------------------------------------------------------------
# --------------------------------------- Sequential Data ---------------------------------
# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------
# --------------------------------------- Sequential Data ---------------------------------
# -----------------------------------------------------------------------------------------

def generate_sequential_data(df, label_col, seq_len):
    """
    Prepares padded/truncated 3D sequential input data for LSTM and TCN models.
    Sequences longer than seq_len are truncated from the left (keeping most recent).
    Sequences shorter than seq_len are zero-padded at the front.
    A binary mask channel is appended as the last feature: 1 = real observation,
    0 = padding.

    Args:
        df (pd.DataFrame): Raw sequential vehicle data.
        label_col (str): Name of the label column.
        seq_len (int): Fixed sequence length.

    Returns:
        tuple: (X, y, vehicle_ids)
            X: shape (n_vehicles, seq_len, n_features + 1)
            y: shape (n_vehicles,)
            vehicle_ids: shape (n_vehicles,)
    """
    sequence_matrix = []
    labels = []
    ids = []

    for vehicle_id, vehicle in df.groupby('vehicle_id'):
        vehicle = vehicle.sort_values('time_step')
        sequence = vehicle.drop(columns=['vehicle_id', label_col, 'time_step']).values
        N = len(sequence)

        mask = np.zeros((seq_len, 1))

        if N >= seq_len:
            sequence = sequence[-seq_len:]
            mask[:] = 1.0
        else:
            padding = np.zeros((seq_len - N, sequence.shape[1]))
            sequence = np.vstack([padding, sequence])
            mask[seq_len - N:] = 1.0

        sequence = np.hstack([sequence, mask])
        sequence_matrix.append(sequence)
        labels.append(vehicle[label_col].iloc[-1])
        ids.append(vehicle_id)

    X = np.stack(sequence_matrix)
    y = np.stack(labels)
    vehicle_ids = np.array(ids)

    return X, y, vehicle_ids


# -----------------------------------------------------------------------------------------
# ---------------------------------- Cost Matrix ------------------------------------------
# -----------------------------------------------------------------------------------------

def compute_total_cost(y_true, y_pred, cost_matrix):
    """
    Computes total cost given true multiclass labels and binary predictions.
    Args:
        y_true: array of actual class labels (0-4)
        y_pred: array of binary predictions (0 or 1)
        cost_matrix: dict mapping (actual_class, predicted_class) to cost
    Returns:
        total_cost: sum of costs for all predictions
    """
    total = 0

    for actual, predicted in zip(y_true, y_pred):
        total += cost_matrix[(actual, predicted)]

    return total

def get_metrics(df, threshold=0.5):
    y_true = df['class_label'].values
    y_pred = (df['probability'].values >= threshold).astype(int)
    # Convert multiclass to binary for standard metrics
    y_true_binary = (y_true > 0).astype(int)
    return {
        'Recall':    recall_score(y_true_binary, y_pred),
        'Precision': precision_score(y_true_binary, y_pred),
        'F1':        f1_score(y_true_binary, y_pred),
        'AUC-ROC':   roc_auc_score(y_true_binary, df['probability'].values),
        'AUC-PR':    average_precision_score(y_true_binary, df['probability'].values),
    }