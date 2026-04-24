import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

# -----------------------------------------------------------------------------------------
# --------------------------------------- Cumulative Features -----------------------------
# -----------------------------------------------------------------------------------------

def get_cumulative_cols(df, cols, threshold=0.95):
    cumulative = []
    for col in cols:
        if df[col].dtype != 'object':
            diffs = df.groupby('vehicle_id')[col].diff().dropna()
            active_diffs = diffs[diffs != 0]

            if len(active_diffs) > 0 and (active_diffs > 0).mean() > threshold:
                cumulative.append(col)
                
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
def generate_sequential_data(df, label_col, seq_len):
    """
    Prepares padded/truncated 3D sequential input data for LSTM and TCN models.
    Sequences longer than seq_len are truncated from the left (keeping most recent).
    Sequences shorter than seq_len are zero-padded at the front.

    Args:
        df (pd.DataFrame): Raw sequential vehicle data.
        label_col (str): Name of the label column.
        seq_len (int): Fixed sequence length.

    Returns:
        tuple: (X, y) where X is shape (n_vehicles, seq_len, n_features) and y is shape (n_vehicles,).
    """
    sequence_matrix = []
    labels = []

    for vehicle_id, vehicle in df.groupby('vehicle_id'):
        sequence = vehicle.drop(columns=['vehicle_id', label_col, 'time_step']).values
        N = len(sequence)

        if N >= seq_len:
            sequence_matrix.append(sequence[-seq_len:])
        else:
            padding = np.zeros((seq_len - N, sequence.shape[1]))
            sequence_matrix.append(np.vstack([padding, sequence]))

        labels.append(vehicle[label_col].iloc[-1])

    X = np.stack(sequence_matrix)
    y = np.stack(labels)

    return X, y


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