import scipy.optimize as optimize
from model import uexe_log_likelihood, uexe_log_pdf
import numpy as np
import matplotlib.pyplot as plt


# 最大似然估计函数
def mle_estimation(log_likelihood_func, initial_guess, data):
    result = optimize.minimize(log_likelihood_func, initial_guess, args=(data,), method='L-BFGS-B',
                                bounds=[(1e-6, None), (1e-6, None)])
    params = result.x
    hessian_inv = result.hess_inv.todense()  # 提取逆Hessian矩阵用于估计标准误差
    se = np.sqrt(np.diag(hessian_inv)) if result.success else [np.nan] * len(params)
    return params, se


data1 = np.array([0.023, 0.032, 0.054, 0.069, 0.081, 0.094, 0.105, 0.127, 0.148,
                 0.169, 0.188, 0.216, 0.255, 0.277, 0.311, 0.361, 0.376, 0.395,
                 0.432, 0.463, 0.481, 0.519, 0.529, 0.567, 0.642, 0.674, 0.752,
                 0.823, 0.887, 0.926])
data2 = np.array([0.853, 0.759, 0.874, 0.800, 0.716, 0.557, 0.503, 0.399, 0.334,
                  0.207, 0.118, 0.097, 0.078, 0.067, 0.056, 0.044, 0.036, 0.026,
                  0.019, 0.014, 0.010, 0.118])
x_values = np.linspace(0, 1, 100)
x_values_add = np.linspace(1e-7, 1-1e-7, 100)
plt.figure(figsize=(10, 5))

# 打乱数据的索引
np.random.seed(42)
shuffled_indices = np.random.permutation(len(data1))

# 划分训练集和验证集
train_size = 250
train_indices = shuffled_indices[:train_size]
val_indices = shuffled_indices[train_size:]
train_data1 = data1[train_indices]
val_data1 = data1[val_indices]
train_data2 = data2[train_indices]
val_data2 = data2[val_indices]

# 使用这些索引来获取训练集和验证集
params, se = mle_estimation(uexe_log_likelihood, [1, 1], train_data1)
pdf_values = np.exp(uexe_log_pdf(params, x_values_add))
plt.subplot(1, 2, 1)
plt.plot(x_values, pdf_values, label='UEXE', color='purple', linestyle='-')
plt.hist(val_data1, bins=np.linspace(0, 1, 10), density=True, alpha=1, color='white', edgecolor='black', label='Val')
plt.hist(train_data1, bins=np.linspace(0, 1, 10), density=True, alpha=0.2, color='gray', edgecolor='gray', label='Train')
plt.xlabel('data1')
plt.ylabel('Estimated PDF')
plt.legend()


plt.subplot(1, 2, 2)

# 使用这些索引来获取训练集和验证集
params, se = mle_estimation(uexe_log_likelihood, [1, 1], train_data2)
pdf_values = np.exp(uexe_log_pdf(params, x_values_add))
plt.plot(x_values, pdf_values, label='UEXE', color='purple', linestyle='-')
plt.hist(val_data2, bins=np.linspace(0, 1, 10), density=True, alpha=1, color='white', edgecolor='black', label='Val')
plt.hist(train_data2, bins=np.linspace(0, 1, 10), density=True, alpha=0.2, color='gray', edgecolor='gray', label='Train')
plt.xlabel('data2')
plt.ylabel('Estimated PDF')
plt.legend()

plt.tight_layout()
plt.show()
