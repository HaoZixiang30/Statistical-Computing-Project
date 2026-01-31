'''
@File: DataGenerator.py
Desc: class DataGenerator genetate data for simulation
'''
from config import parse_args
import math as m
import numpy as np
import random

class DataGenerator:
    def __init__(self, opt):
        self.seed = opt.seed
        self.fun_name = opt.func_name
        np.random.seed(self.seed)
        random.seed(self.seed)

    def __call__(self, K, n_list, delta, lambdaa):
        return self.forward(K, n_list, delta, lambdaa)

    def UExE_func(self, x, delta, lambdaa):
        return 1 - (1 - (delta * lambdaa * m.log(x)) / (delta + lambdaa)) * x**delta
    

    def find_x(self, y, func_name, delta, lam, low=0, high=1, tol=1e-10):
        if func_name == 'UExE':
            while high - low > tol:
                mid = (low + high) / 2.0
                V_mid = self.UExE_func(mid, delta, lam)
                if abs(V_mid - y) < tol: 
                    return mid
                elif V_mid > y: 
                    low = mid
                else:  
                    high = mid
            return (low + high) / 2.0
        else:
            return 0
    
    def inverse_func(self, y_ls, func_name, delta=0, lambdaa=1):
        x_ls = []
        for y in y_ls:
            inverse = self.find_x(y, func_name, delta=delta,lam=lambdaa)
            x_ls.append(inverse)
        return x_ls

    def forward(self, K, n_list, delta=0, lambdaa=1):# eg: K 1000 n_list:[100, 150, 200, 300, 400, 500]
        '''
        @Params: 
        K: The number of the samples
        n_list: The number of each sample
        func_name: The function that we would like to sample
        delta, lambdaa: The parameters of the function

        @Returns:
        samples: dic{K:{n:[sample list]}}:
        '''
        samples = {}
        for i in range(K):
            if i not in samples:
                samples[i] = {}
            for n in n_list:
                random_y = np.random.uniform(0, 1, n)
                random_x = self.inverse_func(random_y, self.fun_name, delta, lambdaa)
                if n not in samples[i]:
                    samples[i][n] = random_x
        return samples
    
if __name__ == '__main__':
    opt = parse_args()
    datagenerator = DataGenerator(opt)
    K = 2
    n_list = [100, 150, 200, 300, 400, 500]
    samples = datagenerator(K, n_list, delta=0.5, lambdaa=0.5)
    print(samples[1][500])