import numpy as np
import scipy.optimize as optimize
from scipy.special import gamma
from scipy import stats
import random
from tqdm import tqdm
from statsmodels.distributions.empirical_distribution import ECDF
import pandas as pd

# 数据集
data1 = np.array([0.023, 0.032, 0.054, 0.069, 0.081, 0.094, 0.105, 0.127, 0.148,
                 0.169, 0.188, 0.216, 0.255, 0.277, 0.311, 0.361, 0.376, 0.395,
                 0.432, 0.463, 0.481, 0.519, 0.529, 0.567, 0.642, 0.674, 0.752,
                 0.823, 0.887, 0.926])
data2 = np.array([0.853, 0.759, 0.874, 0.800, 0.716, 0.557, 0.503, 0.399, 0.334,
                  0.207, 0.118, 0.097, 0.078, 0.067, 0.056, 0.044, 0.036, 0.026,
                  0.019, 0.014, 0.010, 0.118])


def uexe_log_likelihood(params, data):
    delta, lambd = params
    # 检查参数范围
    if delta <= 0 or lambd <= 0:
        return np.inf  # 参数或数据越界时，返回正无穷作为惩罚
    # 计算负对数似然值
    log_pdf = (
        2 * np.log(delta)
        + np.log(1 - lambd * np.log(data))
        - np.log(delta + lambd)
        + (delta - 1) * np.log(data)
    )
    return -np.sum(log_pdf)


def beta_log_likelihood(params, data):
    a, b = params
    if a <= 0 or b <= 0:
        return np.inf
    return -np.sum(stats.beta.logpdf(data, b, a))


def kumaraswamy_log_likelihood(params, data):
    a, b = params
    if a <= 0 or b <= 0:
        return np.inf
    return -np.sum(np.log(b) + np.log(a) + (b - 1) * np.log(data) + (a - 1) * np.log(1 - data**b))


def unit_gamma_log_likelihood(params, data):
    a, b = params
    if a <= 0 or b <= 0:
        return np.inf
    log_pdf = (
        np.log(b)
        + (b-1) * np.log(data)
        + (a-1) * np.log((-b) * np.log(data))
        - np.log(gamma(a))
    )
    return -np.sum(log_pdf)


def power_topp_leone_log_likelihood(params, data):
    a, b = params
    if a <= 0 or b <= 0:
        return np.inf
    return -np.sum(np.log(2) + np.log(a) + np.log(b) + (a * b - 1) * np.log(data) + np.log(1 - data**a) + (b-1) * np.log(2 - data**a))


def bounded_weighted_exponential_log_likelihood(params, data):
    a, b = params
    if a <= 0 or b <= 0:
        return np.inf
    return -np.sum(np.log(a+1) + np.log(b) - np.log(a) + (b-1) * np.log(data) + np.log(1 - data**(a*b)))


def unit_burr_xii_log_likelihood(params, data):
    a, b = params
    if a <= 0 or b <= 0:
        return np.inf
    return -np.sum(np.log(a) + np.log(b) - (1+a) * np.log(1 + (-np.log(data))**b) - np.log(data) - (1-b) * np.log(-np.log(data)))


def unit_birnbaum_saunders_log_likelihood(params, data):
    a, b = params
    if a <= 0 or b <= 0:
        return np.inf
    first = np.sqrt(-b/np.log(data)) + np.sqrt((-b/np.log(data))**3)
    second1 = b/np.log(data)
    second2 = np.log(data)/b
    second = (second1 + second2 + 2)/(a**2)
    second = second/2
    third = 2 * a * b * data * np.sqrt(2*np.pi)
    log_pdf = (
            np.log(first)
            + second
            - np.log(third)
    )
    return -np.sum(log_pdf)


def unit_inverse_gaussian_log_likelihood(params, data):
    a, b = params
    if a <= 0 or b <= 0:
        return np.inf
    second_1 = a * ((np.log(data) + b)**2)
    second_2 = 2 * np.log(data) * (b**2)
    second = second_1 / second_2
    c = (np.log(data))**2
    log_pdf = (
            np.log(a)
            + second
            - 0.5 * np.log(2 * np.pi)
            - np.log(data)
            - np.log(c)
            - 0.5 * np.log(-a/np.log(data))
    )
    return -np.sum(log_pdf)


def unit_mirra_log_likelihood(params, data):
    a, b = params
    if a <= 0 or b <= 0:
        return np.inf
    log_pdf = (
            3 * np.log(b)
            + np.log(a + (a+2) * (data**2) - 2 * a * data)
            + b
            - b/data
            - np.log(2 * (data**4))
            - np.log(a + (b**2))
    )
    return -np.sum(log_pdf)


# 最大似然估计函数
def mle_estimation(log_likelihood_func, initial_guess, data):
    result = optimize.minimize(log_likelihood_func, initial_guess, args=(data,), method='L-BFGS-B',
                                bounds=[(1e-6, None), (1e-6, None)])
    params = result.x
    hessian_inv = result.hess_inv.todense()  # 提取逆Hessian矩阵用于估计标准误差
    se = np.sqrt(np.diag(hessian_inv)) if result.success else [np.nan] * len(params)
    return params, se


# 定义初始猜测值和分布
initial_guesses = {
    'UEXE': [1, 1],
    'Beta': [1, 1],
    'Kumaraswamy': [1, 1],
    'Unit Gamma': [1, 1],
    'Power Topp-Leone': [1, 1],
    'Bounded Weighted Exponential': [1, 1],
    'Unit Burr XII': [1, 1],
    'Unit Birnbaum-Saunders': [1, 1],
    'Unit Inverse Gaussian': [1, 1],
    'Unit Mirra': [1, 1],
}

distributions = {
    'UEXE': uexe_log_likelihood,
    'Beta': beta_log_likelihood,
    'Kumaraswamy': kumaraswamy_log_likelihood,
    'Unit Gamma': unit_gamma_log_likelihood,
    'Power Topp-Leone': power_topp_leone_log_likelihood,
    'Bounded Weighted Exponential': bounded_weighted_exponential_log_likelihood,
    'Unit Burr XII': unit_burr_xii_log_likelihood,
    'Unit Birnbaum-Saunders': unit_birnbaum_saunders_log_likelihood,
    'Unit Inverse Gaussian': unit_inverse_gaussian_log_likelihood,
    'Unit Mirra': unit_mirra_log_likelihood,
}

if __name__ == "__main__":

    M = 1000
    # boot_strap data1
    results_data1 = {}
    for dist_name, log_likelihood_func in distributions.items():
        initial_guess = initial_guesses[dist_name]
        ori_params, _ = mle_estimation(log_likelihood_func, initial_guess, data1)
        boot_out = []
        for _ in range(M):
            data_sample = np.array(random.sample(list(data1), len(data1)))
            params, se = mle_estimation(log_likelihood_func, initial_guess, data_sample)
            boot_out.append(params)
        boot_out = np.array(boot_out)
        
        first_boot = boot_out[:,0]
        sec_boot = boot_out[:,-1]
        
        first_boot_bias = (first_boot - ori_params[0]).mean()
        sec_boot_bias = (sec_boot - ori_params[1]).mean()
        
        first_boot_var = first_boot.var()
        sec_boot_var = sec_boot.var()
        results_data1[dist_name] = {'first_param_bias': first_boot_bias, 'sec_param_bias': sec_boot_bias,
                                    'first_param_var': first_boot_var, 'sec_param_var': sec_boot_var}

    # for k,v in results_data1.items():
    #     print(k,v)

    #     # 转化为 DataFrame
    # df = pd.DataFrame.from_dict(results_data1, orient='index')

    # # 保存为 Excel 文件
    # df.to_excel("data1_bootstrap.xlsx", index=True)

    # boot_strap data2
    results_data2 = {}
    for dist_name, log_likelihood_func in distributions.items():
        initial_guess = initial_guesses[dist_name]
        ori_params, _ = mle_estimation(log_likelihood_func, initial_guess, data2)
        boot_out = []
        for _ in range(M):
            data_sample = np.array(random.sample(list(data2), len(data2)))
            params, se = mle_estimation(log_likelihood_func, initial_guess, data_sample)
            boot_out.append(params)
        boot_out = np.array(boot_out)
        
        first_boot = boot_out[:,0]
        sec_boot = boot_out[:,-1]
        
        first_boot_bias = (first_boot - ori_params[0]).mean()
        sec_boot_bias = (sec_boot - ori_params[1]).mean()
        
        first_boot_var = first_boot.var()
        sec_boot_var = sec_boot.var()
        results_data2[dist_name] = {'first_param_bias': first_boot_bias, 'sec_param_bias': sec_boot_bias,
                                    'first_param_var': first_boot_var, 'sec_param_var': sec_boot_var}

    for k,v in results_data2.items():
        print(k,v)

        # 转化为 DataFrame
    df = pd.DataFrame.from_dict(results_data2, orient='index')

    # 保存为 Excel 文件
    df.to_excel("data2_bootstrap.xlsx", index=True)