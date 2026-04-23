import numpy as np

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
# --------------------------------------- Sequential Data ------------------------------------
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