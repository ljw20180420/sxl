from EnsembleAllocationCode.code_1p.scope_ml import regularized_nmf
from sxl.auROC import load_mouse_data
import pathlib
from sklearn.preprocessing import scale
import numpy as np
import scipy.io as sio

df_f = sio.loadmat("df_f.mat")["df_f"]

A = scale(df_f, axis=1, with_mean=False)
active_test = np.sum(A, axis=1) > 0
A = A[active_test, :]
H, W, best_k = regularized_nmf(A, ds=5, tol=2)
H = H / (np.sum(H, axis=1)[:, None])
