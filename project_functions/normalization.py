import numpy as np
from sklearn.preprocessing import MaxAbsScaler

class MaxAbsLogScaler(MaxAbsScaler):

    """
    Scaling with logarithmic function log(x+1) and MaxAbs normalization (by maximum absolute value).
    Adapted for time-series data formatted as 2-dimensional matrix (samples, dates) 
    """
    
    def __init__(self):
        super().__init__()

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        return super().fit(np.log1p(X).T, y)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return super().transform(np.log1p(X).T).T
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return np.expm1(
            super().inverse_transform(X = X.T)
            ).T