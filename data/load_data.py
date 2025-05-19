import sys
import os
import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_medians():
    jfunc_all_data = xr.load_dataset("./data/jfunction_data_med.nc")
    X_raw = jfunc_all_data.v.to_numpy()
    X_raw_0 = X_raw[0, ~np.any(np.isnan(X_raw[0]), axis=(1,2))]
    X_raw_1 = X_raw[1, ~np.any(np.isnan(X_raw[1]), axis=(1,2))]
    X_raw_2 = X_raw[2, ~np.any(np.isnan(X_raw[2]), axis=(1,2))]
    X = np.concatenate([X_raw_0, X_raw_1, X_raw_2])
    X = X.transpose((0,2,1))
    y = np.concatenate([np.zeros(X_raw_0.shape[0], dtype=int), np.ones(X_raw_1.shape[0], dtype=int), np.full(X_raw_2.shape[0], 2)]).reshape((-1, 1))
    return X, y


def load_ensembles():
    jfunc_all_data = xr.load_dataset("./data/jfunction_data_ensemble.nc")
    X_raw = jfunc_all_data.v.to_numpy()
    X_raw_0 = X_raw[0, ~np.any(np.isnan(X_raw[0]), axis=(1,2))]
    X_raw_1 = X_raw[1, ~np.any(np.isnan(X_raw[1]), axis=(1,2))]
    X_raw_2 = X_raw[2, ~np.any(np.isnan(X_raw[2]), axis=(1,2))]
    X = np.concatenate([X_raw_0, X_raw_1, X_raw_2])
    X = X.transpose((0,2,1))
    y = np.concatenate([np.zeros(X_raw_0.shape[0], dtype=int), np.ones(X_raw_1.shape[0], dtype=int), np.full(X_raw_2.shape[0], 2)]).reshape((-1, 1))
    return X, y

def load_medians_clean():
    return np.load("./data/jfunction_data_med_pred.npy"), np.load("./data/jfunction_data_med_target.npy"), np.load("./data/jfunction_data_med_filter.npy")
