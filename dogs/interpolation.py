import numpy as np
from scipy import optimize


class InterParams:
    def __init__(self, xE):
        self.method = "NPS"
        self.n  = xE.shape[0]
        self.m  = xE.shape[1]
        self.xi = np.copy(xE)
        self.w  = None
        self.v  = None
        self.y  = None
        self.sigma = None

    def interpolateparameterization(self, yE):
        self.w  = np.zeros((self.m, 1))
        self.v  = np.zeros((self.n + 1, 1))
        self.y  = np.copy(yE)
        if self.method == 'NPS':
            A = np.zeros((self.m, self.m))
            for ii in range(self.m):
                for jj in range(self.m):
                    A[ii, jj] = (np.dot(self.xi[:, ii] - self.xi[:, jj], self.xi[:, ii] - self.xi[:, jj])) ** (3.0 / 2.0)

            V = np.vstack((np.ones((1, self.m)), self.xi))
            A1 = np.hstack((A, np.transpose(V)))
            A2 = np.hstack((V, np.zeros((self.n + 1, self.n + 1))))
            yi = self.y[np.newaxis, :]
            b = np.concatenate([np.transpose(yi), np.zeros((self.n + 1, 1))])
            A = np.vstack((A1, A2))
            wv = np.linalg.lstsq(A, b, rcond=-1)
            wv = np.copy(wv[0])
            self.w = np.copy(wv[:self.m])
            self.v = np.copy(wv[self.m:])
            yp = np.zeros(self.m)
            for ii in range(self.m):
                yp[ii] = self.inter_val(self.xi[:, ii])
            return yp

    def regressionparameterization(self, yi, sigma):
        # Notice xi, yi and sigma must be a two dimension matrix, even if you want it to be a vector.
        # or there will be error
        self.y = np.copy(yi)
        self.sigma = np.copy(sigma)
        if self.method == 'NPS':
            A = np.zeros((self.m, self.m))
            for ii in range(self.m):  # for ii =0 to m-1 with step 1; range(1,N,1)
                for jj in range(self.m):
                    A[ii, jj] = (np.dot(self.xi[:, ii] - self.xi[:, jj], self.xi[:, ii] - self.xi[:, jj])) ** (3.0 / 2.0)
            V = np.concatenate((np.ones((1, self.m)), self.xi), axis=0)
            w1 = np.linalg.lstsq(np.dot(np.diag(1 / sigma), V.T), (yi / sigma).reshape(-1, 1), rcond=None)
            w1 = np.copy(w1[0])
            b = np.mean(np.divide(np.dot(V.T, w1) - yi.reshape(-1, 1), sigma.reshape(-1, 1)) ** 2)
            wv = np.zeros([self.m + self.n + 1])
            if b < 1:
                wv[self.m:] = np.copy(w1.T)
                rho = 1000
                wv = np.copy(wv.reshape(-1, 1))
            else:
                rho = 1.1
                fun = lambda rho: self.smoothing_polyharmonic(rho, A, V)[0]
                rho = optimize.fsolve(fun, rho)
                b, db, wv = self.smoothing_polyharmonic(rho, A, V)
            self.w = np.copy(wv[:self.m])
            self.v = np.copy(wv[self.m:self.m + self.n + 1])
            yp = np.zeros(self.m)
            while (1):
                for ii in range(self.m):
                    yp[ii] = self.inter_val(self.xi[:, ii])
                residual = np.max(np.divide(np.abs(yp - yi), sigma))
                if residual < 2:
                    break
                rho *= 0.9
                b, db, wv = self.smoothing_polyharmonic(rho, A, V)
                self.w = np.copy(wv[:self.m])
                self.v = wv[self.m:self.m + self.n + 1]
        return yp

    def smoothing_polyharmonic(self, rho, A, V):
        A01 = np.concatenate((A + rho * np.diag(self.sigma ** 2), np.transpose(V)), axis=1)
        A02 = np.concatenate((V, np.zeros(shape=(self.n + 1, self.n + 1))), axis=1)
        A1 = np.concatenate((A01, A02), axis=0)
        b1 = np.concatenate([self.y.reshape(-1, 1), np.zeros(shape=(self.n + 1, 1))])
        wv = np.linalg.lstsq(A1, b1, rcond=None)
        wv = np.copy(wv[0])
        b = np.mean(np.multiply(wv[:self.m], self.sigma.reshape(-1, 1)) ** 2 * rho ** 2) - 1
        bdwv = np.concatenate([np.multiply(wv[:self.m], self.sigma.reshape(-1, 1) ** 2), np.zeros((self.n + 1, 1))])
        Dwv = np.linalg.lstsq(-A1, bdwv, rcond=None)
        Dwv = np.copy(Dwv[0])
        db = 2 * np.mean(np.multiply(wv[:self.m] ** 2 * rho + rho ** 2 * np.multiply(wv[:self.m], Dwv[:self.m]), self.sigma ** 2))
        return b, db, wv

    def inter_val(self, x):
        '''
        Calculate the value of objective interpolant at x.
        :param x:           The intended position to calculate the gradient of interpolation/regression function
        :param inter_par:   The parameter set for interpolation/regression
        return:             The interpolation/regression function values at x.
        '''
        x = x.reshape(-1, 1)
        if self.method == "NPS":
            S = self.xi - x
            return np.dot(self.v.T, np.concatenate([np.ones((1, 1)), x], axis=0)) + np.dot(self.w.T, (
                np.sqrt(np.diag(np.dot(S.T, S))) ** 3))

    def inter_grad(self, x):
        '''
        Calculate the gradient of interpolant at x.
        :param x:           The intended position to calculate the gradient of interpolation/regression function
        :param inter_par:   The parameter set for interpolation/regression
        return:             The column vector of the gradient information at point x.
        '''

        x = x.reshape(-1, 1)
        g = np.zeros((self.n, 1))
        if self.method == "NPS":
            for ii in range(self.m):
                X = x - self.xi[:, ii].reshape(-1, 1)
                g = g + 3 * self.w[ii] * X * np.linalg.norm(X)
            g = g + self.v[1:]
            return g

    def interpolate_hessian(self, x):
        '''
        :param x:           The intended position to calculate the gradient of interpolation/regression function
        :param inter_par:   The parameter set for interpolation/regression
        return:             The hessian matrix of at x.
        '''
        if self.method == "NPS":
            H = np.zeros((self.n, self.n))
            for ii in range(self.m):
                X = x[:, 0] - self.xi[:, ii]
                if np.linalg.norm(X) > 1e-5:
                    H = H + 3 * self.w[ii] * ((X * X.T) / np.linalg.norm(X) + np.linalg.norm(X) * np.identity(self.n))
            return H

    def inter_cost(self, x):
        x = x.reshape(-1, 1)
        M = self.inter_val(x)
        DM = self.inter_grad(x)
        return M, DM.T
