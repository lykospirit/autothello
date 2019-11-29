import numpy as np

def compute_mse(y_true, y_pred):
  return np.mean(np.square(y_true - y_pred))

def compute_sse(y_true, y_pred):
  return np.sum(np.square(y_true - y_pred))

# epsilon clipping from https://stackoverflow.com/questions/47377222/cross-entropy-function-python
def compute_cat_xentropy(y_true, y_pred, epsilon=1e-9):
  return -np.sum(y_true * np.log(np.clip(y_pred, epsilon, 1.0-epsilon)))
