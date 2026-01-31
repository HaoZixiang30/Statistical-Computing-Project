'''
@File: MLE_Estimate.py
Desc: class Maximum Likelihood Estimation for delta and lambda
'''
import numpy as np

class MLE_Estimate:
    def __init__(self, opt):
        self.delta_init = opt.delta_init
        self.lambda_init = opt.lambda_init

    def __call__(self, x, tol=1e-7, max_iter=1000):
        return self.forward(x, tol, max_iter)

    def gradient(self, delta, lambdaa, x):
        n = len(x)
        grad_delta = (2 * n / delta) - (n / (delta + lambdaa)) + np.sum(np.log(x))
        grad_lambd = -(n / (delta + lambdaa)) - np.sum(np.log(x) / (1 - lambdaa * np.log(x)))

        return np.array([grad_delta, grad_lambd])

    def hessian(self, delta, lambdaa, x):
        n = len(x)
        h11 = -2 * n / (delta ** 2) + n / ((delta + lambdaa) ** 2)
        h22 = n / ((delta + lambdaa) ** 2) - np.sum((np.log(x) ** 2) / ((1 - lambdaa * np.log(x)) ** 2))
        h12 = n / ((delta + lambdaa) ** 2)
        H = np.array([[h11, h12], [h12, h22]])

        return H

    def forward(self, x, tol=1e-7, max_iter=1000):
        delta, lambdaa = self.delta_init, self.lambda_init
        for i in range(max_iter):
            grad = self.gradient(delta, lambdaa, x)
            H = self.hessian(delta, lambdaa, x)
            try:
                H_inv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                return self.delta_init + np.random.normal(0, 0.05), self.lambda_init + np.random.normal(0, 0.05)
            update = H_inv @ grad
            delta, lambdaa = np.array([delta, lambdaa]) - update

            if np.linalg.norm(grad) < tol:
                break
        delta = delta.clip(0.25*self.delta_init+ np.random.normal(0, 0.05), 2*self.delta_init+ np.random.normal(0, 0.05))
        lambdaa = lambdaa.clip(0.25*self.lambda_init+ np.random.normal(0, 0.05), 2*self.lambda_init+ np.random.normal(0, 0.05))
        return delta, lambdaa
    

