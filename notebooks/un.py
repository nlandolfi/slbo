import abc
import numpy as np

class Model(abc.ABC):
    def __init__(self, d):
        self.d = d

    def impute_nan(self, x):
        return self.impute(x, Kfromnans(x))

    @abc.abstractmethod
    def impute(self, x, K):
        pass

class CovarianceModel(Model):
    def __init__(self, d):
        super().__init__(d)
        self.S = None

    def train(self, Xtrain):
        assert Xtrain.shape[1] == self.d
        self.S = empirical_covariance(Xtrain)

    def impute(self, x, K):
        return conditional_mean(self.S, x, K)

class L2ConstantModel(Model):
    def __init__(self, d):
        super().__init__(d)
        self.theta = None

    def train(self, Xtrain):
        assert Xtrain.shape[1] == self.d
        self.theta = np.mean(Xtrain, axis=0)

    def impute(self, x, K):
        return impute_constant(self.theta, x, K)

class L1ConstantModel(Model):
    def __init__(self, d):
        super().__init__(d)
        self.theta = None

    def train(self, Xtrain):
        assert Xtrain.shape[1] == self.d
        self.theta = np.median(Xtrain, axis=0)

    def impute(self, x, K):
        return impute_constant(self.theta, x, K)

class KMeansModel(Model):
    def __init__(self, d, k):
        super().__init__(d)
        self.k = k
        self.thetas = None

    def train(self, Xtrain):
        assert Xtrain.shape[1] == self.d
        _, thetas, _ = k_means(Xtrain, self.k)
        self.thetas = thetas

    def impute(self, x, K):
        return impute_kmeans(self.thetas, x, K)

def Kfromnans(x):
    K = []
    for i in range(len(x)):
        if not np.isnan(x[i]):
            K.append(i)
    return K

def empirical_covariance(X):
    n = X.shape[0]
    S = 0.
    for i in range(n):
        S += np.outer(X[i, :], X[i, :])
    S = S / float(n)
    return S

def conditional_mean(S, xmatch, K):
    d = S.shape[0]
    xa = np.zeros(len(K))
    
    for (s, k) in enumerate(K):
        xa[s] = xmatch[k]
    
    SAA = np.zeros((len(K), len(K)))
    SBA = np.zeros((d-len(K), len(K)))
    
    for (s, i) in enumerate(K, ):
        for (t, j) in enumerate(K):
            SAA[s, t] = S[i, j]
    
    sk = set(K)
    U = []
    for i in range(d):
        if i not in sk:
            U.append(i)
        
    for (s, i) in enumerate(U):
        for (t, j) in enumerate(K):
            SBA[s, t] = S[i, j]
            
    guessb = np.matmul(SBA, np.matmul(np.linalg.inv(SAA), xa))

    xout = xmatch.copy()
    for (s, u) in enumerate(U):
        xout[u] = guessb[s]
   
    return xout

def impute_constant(rep, xmatch, K):
    out = rep.copy()
    for k in K:
        out[k] = xmatch[k]
    return out

def k_means(X, k, iters=20):
    n, d = X.shape
    c = np.random.randint(k, size=n)
    thetas = np.zeros((d, k))
    losses = []
    for i in range(iters):
        for j in range(k):
            thetas[:, j] = np.mean(X[c == j], axis=0)
        mindiffs = []
        for l in range(n):
            diffs = [np.linalg.norm(X[l, :] - thetas[:, j]) for j in range(k)]
            mindiff, minindex = np.min(diffs), np.argmin(diffs)
            c[l] = minindex
            mindiffs.append(mindiff)
        losses.append(np.mean(mindiffs))
    return (c, thetas, losses)

def impute_kmeans(thetas, xmatch, K):
    nmeans = thetas.shape[1]
    theta = thetas[:, np.argmin([np.sum(np.square([thetas[:, i][k] - xmatch[k] for k in K])) for i in range(nmeans)])]
    out = theta.copy()
    for i in K:
        out[i] = xmatch[i]
    return out