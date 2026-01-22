import numpy as np
import scipy as sp
from scipy.spatial import distance
np.random.seed(3633914)
class OnlineKernelEDMD:
    def __init__(self, X, Y, kernel="gaussian", **kwargs):
        """
        X: N_dim * M_data
        Y: N_dim * M_data
        """
        # For updating, this class needs to preserve previous data.
        self.X = X
        self.Y = Y
        self.kernel = kernel
        
        if self.kernel == "gaussian":
            assert "eps" in kwargs, "gaussian kernel needs eps param"
            self.eps = kwargs["eps"]
            self.f = self.gaussian
        elif self.kernel == "polynomial":
            assert ("p" in kwargs) and ("c" in kwargs), "polynomial kernel needs c and p param"
            self.p = kwargs["p"]
            self.c = kwargs["c"]
            if "gamma" in kwargs:
                self.gamma = kwargs["gamma"]
            else:
                self.gamma = 1.0
            self.f = self.polynomial
        self.G = self.f(self.X, self.X)
        self.A = self.f(self.X, self.Y).T
        
    def fit(self, thr=0.0):
        # 低ランク近似
        self.s, self.U = sp.linalg.eigh(self.G)
        # 固有値を昇順にソートし，固有ベクトルをそれに対応させる
        idx = self.s.argsort()
        self.s = self.s[idx]
        self.U = self.U[:, idx]
        self.s, mask = self.low_rank_approximation(self.s, thr)
        if self.s.shape[0] == 0: # 全部捨てられたらエラー
            raise ValueError("low_rank_approximation removed all modes")
        self.U = self.U[:, mask]
        S_inv = np.diag(1/self.s)
        self.G = self.U @ np.diag(self.s) @ self.U.T
        self.K = self.U @ S_inv @ self.U.T @ self.A
        self.calc_modes(S_inv)
        return None
        
    def calc_modes(self, S_inv):
        # 固有値,固有ベクトルを求めてモードを求める
        _K = S_inv @ self.U.T @ self.A @ self.U
        self.eigenvalues, V = np.linalg.eig(_K)
        self.eigenvectors = self.U @ V
        self.modes = sp.linalg.pinv(V) @ S_inv @ self.U.T @ self.X.T
        return None

    def calc_eigenfunction(self, x):
        """
        x: N_dim * 1
        """
        #次の座標を予測したいデータ点を入力とする固有関数の値を求める
        if self.kernel == "gaussian":
            phi_Xx = np.exp(-np.linalg.norm(self.X - x.reshape(-1,1), axis=0)**2/(2*(self.eps**2))) @ self.eigenvectors
        elif self.kernel == "polynomial":
            phi_Xx = (self.c + x.T @ self.X)**self.p @ self.eigenvectors
        return phi_Xx
    def predict(self, x):
        """
        x: N_dim * 1
        return: N_dim * 1
        """
        phi_Xx = self.calc_eigenfunction(x)
        return np.real((self.eigenvalues * phi_Xx.flatten()) @ self.modes)

    def gaussian(self, X, Y):
        """
        X: N_dim * M_data
        Y: N_dim * M_data
        return Gram matrix, ij element = k(x_j, y_i)
        """
        diff = X[:, :, None] - Y[:, None, :]     # (n, M, M) ← Cが生成
        d2   = np.einsum('ijk,ijk->jk', diff, diff)   # 距離² : (M, M)
        G    = np.exp(-0.5 * d2 / self.eps**2)            # 要素一括 exp

        return G
    def polynomial(self, X, Y):
        """
        c: param
        d: order
        X: N_dim * M_data
        Y: N_dim * M_data
        return Gram matrix, each element = k(x_i, y_i)
        """
        dots = X.T @ Y                      # (M, M)  ← BLAS dgemm
        G    = (self.gamma * dots + self.c)**self.p
        return G

    def eig_update(self, c, b, thr):
        """
        Only for gramian
        Q: Matrix of eigenvectors
        a: col vector
        b: col vector
        """
        r = self.s.shape[0]
        
        A = np.hstack((c, b)) # (M,2)
        D = np.array([[0, 1],[1, 0]]) # (2,2)
        R12   = self.U.T @ A                 # (r,2)   射影   O(M·r)
        A_res = A - self.U @ R12             # (M,2)   残差   O(M·r)

        Q2, R22 = np.linalg.qr(A_res, mode="reduced")   # Q2:(M,2)  R22:(2,2)

        U = np.hstack((self.U, Q2)) # (M,r+2)

        R_top = np.hstack((np.eye(r), R12))
        R_bot = np.hstack((np.zeros((2, r)), R22))
        R     = np.vstack((R_top, R_bot)) # (r+2,r+2)
    
        K = np.zeros((r+2, r+2))
        K[:r,:r] += np.diag(self.s)
        K[r:, r:]+= D
        K = R @ K @ R.T
        s_prime, U_prime = sp.linalg.eigh(K)
        idx = s_prime.argsort()
        s_prime = s_prime[idx]
        U_prime = U_prime[:, idx]
        s_prime, mask = self.low_rank_approximation(s_prime, thr)
        U_updated = U @ U_prime
        U_updated = U_updated[:, mask]
        
        return s_prime, U_updated 
    
    def window_update(self, new_x, new_y, idx=0, thr=0.0):
        """
        Description:
            swap old data and new data from gramian G and covariance matrix A
            If swap_index = 0, swap oldest data. In other words, swap_index=0 means Sliding Window Update.
        Args:
            new_x, new_y: column vector(N_dim * 1)
        Return:
            updated singlar value of G, updated eigenvector of G
        """
        # swap old data and new data from Data matrix
        self.X[:,idx] = new_x
        self.Y[:,idx] = new_y
        # calc kernel vector, k_ij = k(x_i, x_new)

        if self.kernel == "gaussian":
            diff = self.X - new_x[:,None]                          # (n,M)
            d2   = np.einsum('ij,ij->j', diff, diff)          # (M,)
            d = np.exp(-0.5*d2/self.eps**2)                    # (M,)
            a_col = np.exp(-0.5 * np.einsum('ij,ij->j',
                                            self.Y - new_x[:,None],
                                            self.Y - new_x[:,None]) / self.eps**2)
            a_row = np.exp(-0.5 * np.einsum('ij,ij->j',
                                            self.X - new_y[:,None],
                                            self.X - new_y[:,None]) / self.eps**2)
            
        elif self.kernel == "polynomial":
            d = (self.c + self.gamma * new_x.reshape(1, -1) @ self.X)**self.p # x_2 ~ x_M+1
            a_col = (self.gamma * (self.Y.T @ new_x) + self.c)**self.p
            a_row = (self.gamma * (self.X.T @ new_y) + self.c)**self.p
        else:
            raise ValueError("unknown kernel")
        self.A[:, idx] = a_col      # 列を置換
        self.A[idx, :] = a_row      # 行を置換
            
        # swap an arbitary data to a new data point
        c = self.G[:, idx].reshape(-1,1)
        d = d.reshape(-1, 1)
        c = d-c
        c[idx] /= 2
        b = np.zeros(self.U.shape[0]).reshape(-1,1)
        b[idx] += 1

        # update eigenvalue and eigenvector
        self.s, self.U = self.eig_update(c, b, thr)
        S_inv = np.diag(1/self.s)
        self.calc_modes(S_inv)
        self.G = self.U @ np.diag(self.s) @ self.U.T
        return self.s, self.U

    def low_rank_approximation(self, s: np.ndarray, thr: float):
        s = s.copy()
        s[s <= 1e-12] = 0.0               # ゼロに丸める
        nz_idx = np.flatnonzero(s)        # ★ ゼロ以外の位置
        if thr == 0.0:                    # そのまま全部返す
            return s[nz_idx], nz_idx
    
        # 以下は thr > 0 の従来ロジック
        if thr >= 1:
            k = min(int(thr), nz_idx.size)
            keep = nz_idx[np.argsort(s[nz_idx])[-k:]]
        else:
            cs = np.cumsum(s[nz_idx])
            keep = nz_idx[np.where(cs / cs[-1] >= thr)[0]]
        return s[keep], keep
if __name__ == "__main__":
    X = np.array([[1, 1, 3],
                 [1, 2, 5]])
    Y = np.array([[1, 3, 4],
                 [2, 5, 6]])
    x = np.array([[1], 
                  [1]])
    #kedmd = kEDMD(X, Y, kernel="polynominal", c=0.1, p=5)
    kedmd = kEDMD(X, Y, kernel="gaussian", eps=2)
    kedmd.fit()
    print(kedmd.predict(x))