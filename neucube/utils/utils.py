import torch
import numpy as np
import torch.nn.functional as F

def print_summary(info_ls):
    """
    Prints a summary of information in a tabular format.

    Args:
        info_ls (list): A list of lists representing the information to be summarized.
                        Each inner list represents a row of information.

    Example:
        info = [
            ['Name', 'Age', 'Location'],
            ['John Doe', '30', 'New York'],
            ['Jane Smith', '25', 'San Francisco']
        ]
        print_summary(info)

    Output:
        Name        Age   Location
        John Doe    30    New York
        Jane Smith  25    San Francisco
    """
    widths = [max(map(len, col)) for col in zip(*info_ls)]
    for row in info_ls:
        print("  ".join((val.ljust(width) for val, width in zip(row, widths))))

def SNR(Xin, yin):
    Xin = torch.tensor(Xin, dtype=torch.float32)
    yin = torch.tensor(yin, dtype=torch.int64)
    
    classes = torch.unique(yin)
    means = []
    stds = []
    
    for cls in classes:
        X_cls = Xin[torch.where(yin == cls)]
        means.append(torch.mean(X_cls, axis=0))
        stds.append(torch.std(X_cls, axis=0, unbiased=True))
    
    means = torch.stack(means)
    stds = torch.stack(stds)
    
    # Compute the SNR for each feature
    mean_diff = torch.max(means, dim=0).values - torch.min(means, dim=0).values
    std_sum = torch.sum(stds, dim=0)
    ratios = torch.abs(mean_diff) / std_sum
    
    return ratios

def linear_interpolation(array, num_points):
    array = array.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    array = F.interpolate(array, size=num_points, mode='linear', align_corners=True)
    return array.squeeze()

def interpolate(_data_, num_points=52):
    data_interpol = []
    for arr_ in _data_:
        arr_ = torch.tensor(arr_, dtype=torch.float32)
        samp = torch.zeros((num_points, arr_.shape[-1]), dtype=torch.float32)
        for i in range(arr_.shape[-1]):
            arr_interpol = linear_interpolation(arr_[:, i], num_points)
            samp[:, i] = arr_interpol
        data_interpol.append(samp)
    return torch.stack(data_interpol)

def SeparationIndex(states, labels):
    classes = labels.unique()
    N = len(classes)
    X_cl = []
    for cl in classes:
        X_cl.append(states[labels == cl])
    c_means = torch.stack([i.mean(axis=0) for i in X_cl])
    cd = torch.cdist(c_means, c_means, p=2)/N**2
    mask = torch.triu(torch.ones_like(cd, dtype=torch.bool), diagonal=1)
    cd = cd[mask].sum()
    cv = torch.stack([torch.cdist(j.unsqueeze(0), i, p=2).sum()/N for i,j in zip(X_cl, c_means)]).sum()/N
    return cd/(cv+1)