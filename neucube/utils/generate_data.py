import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def generate_complex_trends(n_timesteps, n_features, trend_type='mixed', randomness=0.1, seed=None):
    """
    Generate complex trends for features over time with added randomness.

    Parameters:
    - n_timesteps: int, number of time steps
    - n_features: int, number of features
    - trend_type: str, type of trend ('sin', 'exp', 'poly', or 'mixed')
    - randomness: float, level of randomness to add to the trends
    - seed: int or None, random seed for reproducibility

    Returns:
    - trends: np.ndarray, generated trends
    """
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(None)

    time = np.linspace(0, 1, n_timesteps)
    trends = np.zeros((n_timesteps, n_features))

    for i in range(n_features):
        if trend_type == 'sin':
            trends[:, i] = np.sin(2 * np.pi * (i + 1) * time)
        elif trend_type == 'exp':
            trends[:, i] = np.exp(time * (i + 1))
        elif trend_type == 'poly':
            trends[:, i] = time ** (i + 1)
        elif trend_type == 'mixed':
            trends[:, i] = (np.sin(2 * np.pi * (i + 1) * time) + 
                            np.exp(time * (i + 1)) - 
                            time ** (i + 1))
        else:
            raise ValueError("Unknown trend type. Choose from 'sin', 'exp', 'poly', or 'mixed'.")
        
        trends[:, i] += randomness * np.random.randn(n_timesteps)
    
    return trends

def generate_longitudinal_data(n_samples=1000, n_timesteps=10, n_features=5, n_redundant_features=3, class_sep=1.0, trend_type='mixed', randomness=0.1, seed=None):
    """
    Generate longitudinal classification data with complex trends, time dependencies, and added randomness.

    Parameters:
    - n_samples: int, number of samples
    - n_timesteps: int, number of time steps
    - n_features: int, number of relevant features
    - n_redundant_features: int, number of redundant features
    - class_sep: float, parameter to adjust the separability of classes
    - trend_type: str, type of trend ('sin', 'exp', 'poly', or 'mixed')
    - randomness: float, level of randomness to add to the trends
    - seed: int or None, random seed for reproducibility

    Returns:
    - data: pd.DataFrame, generated data
    """
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(None)

    data = []

    class_trends = {
        0: generate_complex_trends(n_timesteps, n_features, trend_type, randomness, seed),
        1: generate_complex_trends(n_timesteps, n_features, trend_type, randomness, seed)
    }

    for i in range(n_samples):
        class_label = np.random.choice([0, 1])
        
        base_signal = np.random.randn(n_features)
        
        class_offset = class_label * class_sep
        
        time_series = []

        for t in range(n_timesteps):
            trend = class_trends[class_label][t, :]
            noise = np.random.randn(n_features) * randomness
            time_features = base_signal + class_offset + trend + noise
            
            redundant_features = np.random.randn(n_redundant_features)
            combined_features = np.concatenate([time_features, redundant_features])
            time_series.append(combined_features)

        sample_data = pd.DataFrame(time_series, columns=[f'feature_{j}' for j in range(n_features + n_redundant_features)])
        sample_data['time'] = range(n_timesteps)
        sample_data['class'] = class_label
        sample_data['sample_id'] = i
        data.append(sample_data)

    data = pd.concat(data, ignore_index=True)    
    return data