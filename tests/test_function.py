import sys
from pathlib import Path
import pandas as pd
import pytest
import numpy as np
import torch

# Add root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import utils

def test_prepare_lr_data_logic():
    # Creating a sample DataFrame to test the function
    df = pd.DataFrame({
        'vehicle_id': [1, 1, 2, 2],
        'time_step':  [1, 2, 1, 2],
        'feature_A':  [10, 20, 30, 40], # We want 20 and 40
        'label':      [0, 0, 1, 1]
    })
    
    # Run the function
    X, y = utils.prepare_lr_data(df, 'label')
    
    # Assess the results
    assert len(X) == 2, "Should only have one row per vehicle"
    assert X.loc[1, 'feature_A'] == 20, "Should have grabbed the LAST timestep for vehicle 1"
    assert X.loc[2, 'feature_A'] == 40, "Should have grabbed the LAST timestep for vehicle 2"
    assert 'time_step' not in X.columns, "Function failed to drop time_step column"
    assert len(y) == 2