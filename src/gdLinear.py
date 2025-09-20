import numpy as np

class GDLinearReg:
    def __init__(self, lr=0.01, n_iter=1000, fit_intercept=True):
        self.lr = lr
        self.n_iter = n_iter
        self.fit_intercept = fit_intercept
        self.history_ = []  # for å lagre cost (MSE) per iterasjon
    
    def _add_bias(self, X):
        if self.fit_intercept:
            return np.c_[np.ones((X.shape[0], 1)), X]
        return X
    
    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).reshape(-1, 1)
        Xb = self._add_bias(X)

        # init weights
        self.w = np.zeros((Xb.shape[1], 1))

        for _ in range(self.n_iter):
            y_pred = Xb @ self.w
            error = y_pred - y
            grad = (2/len(Xb)) * Xb.T @ error
            self.w -= self.lr * grad

            cost = np.mean(error**2)
            self.history_.append(cost)
        return self
    
    def predict(self, X):
        X = np.array(X, dtype=float)
        Xb = self._add_bias(X)
        return (Xb @ self.w).ravel()
    
    @property
    def coef_(self):
        return self.w[1:].ravel() if self.fit_intercept else self.w.ravel()
    
    @property
    def intercept_(self):
        return float(self.w[0]) if self.fit_intercept else 0.0
