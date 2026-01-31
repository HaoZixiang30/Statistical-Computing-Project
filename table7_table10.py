import numpy as np
import scipy.optimize as optimize
from scipy.special import gamma
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF

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

# 计算每种分布的参数估计和标准误差（对于数据集1）
results_data1 = {}
for dist_name, log_likelihood_func in distributions.items():
    initial_guess = initial_guesses[dist_name]
    params, se = mle_estimation(log_likelihood_func, initial_guess, data1)
    results_data1[dist_name] = {'params': params, 'se': se}

# 计算每种分布的参数估计和标准误差（对于数据集2）
results_data2 = {}
for dist_name, log_likelihood_func in distributions.items():
    initial_guess = initial_guesses[dist_name]
    params, se = mle_estimation(log_likelihood_func, initial_guess, data2)
    results_data2[dist_name] = {'params': params, 'se': se}


# 打印结果的函数
def format_results(results):
    print("MLE estimates and their SEs are in parentheses for the dataset.")
    print(f"{'Model':<15}{'MLE Estimates':<50}")
    for dist_name, result in results.items():
        params = result['params']
        ses = result['se']
        if dist_name == "UEXE":
            print(f"{dist_name}(λ,δ)".ljust(15) +
                  f"̂λ={params[0]:.6f}({ses[0]:.6f})".ljust(30) +
                  f"̂δ={params[1]:.6f}({ses[1]:.6f})")
        elif dist_name == "Beta":
            print(f"{dist_name}(κ,γ)".ljust(15) +
                  f"̂κ={params[0]:.6f}({ses[0]:.6f})".ljust(30) +
                  f"̂γ={params[1]:.6f}({ses[1]:.6f})")
        elif dist_name == "Kumaraswamy":
            print(f"Kum(a,b)".ljust(15) +
                  f"̂a={params[0]:.6f}({ses[0]:.6f})".ljust(30) +
                  f"̂b={params[1]:.6f}({ses[1]:.6f})")
        elif dist_name == "Unit Gamma":
            print(f"UG(θ,ρ)".ljust(15) +
                  f"̂θ={params[0]:.6f}({ses[0]:.6f})".ljust(30) +
                  f"ρ={params[1]:.6f}({ses[1]:.6f})")
        elif dist_name == "Power Topp-Leone":
            print(f"PTL(ν,τ)".ljust(15) +
                  f"̂ν={params[0]:.6f}({ses[0]:.6f})".ljust(30) +
                  f"̂τ={params[1]:.6f}({ses[1]:.6f})")
        elif dist_name == "Bounded Weighted Exponential":
            print(f"BWE(σ,η)".ljust(15) +
                  f"̂σ={params[0]:.6f}({ses[0]:.6f})".ljust(30) +
                  f"̂η={params[1]:.6f}({ses[1]:.6f})")
        elif dist_name == "Unit Burr XII":
            print(f"UBXII(μ,ϕ)".ljust(15) +
                  f"̂μ={params[0]:.6f}({ses[0]:.6f})".ljust(30) +
                  f"̂ϕ={params[1]:.6f}({ses[1]:.6f})")
        elif dist_name == "Unit Birnbaum-Saunders":
            print(f"UBS(α,β)".ljust(15) +
                  f"̂α={params[0]:.6f}({ses[0]:.6f})".ljust(30) +
                  f"̂β={params[1]:.6f}({ses[1]:.6f})")
        elif dist_name == "Unit Inverse Gaussian":
            print(f"UIG(ζ,ω)".ljust(15) +
                  f"̂ζ={params[0]:.6f}({ses[0]:.6f})".ljust(30) +
                  f"̂ω={params[1]:.6f}({ses[1]:.6f})")
        elif dist_name == "Unit Mirra":
            print(f"UM(ψ,Θ)".ljust(15) +
                  f"̂ψ={params[0]:.6f}({ses[0]:.6f})".ljust(30) +
                  f"̂Θ={params[1]:.6f}({ses[1]:.6f})")


# 打印数据集1结果
format_results(results_data1)
print()
# 打印数据集2结果
format_results(results_data2)
print()


# 计算信息准则：AIC、CAIC、HQIC
def calculate_info_criteria(log_likelihood, k, n):
    aic = 2 * k - 2 * log_likelihood
    caic = aic + 2 * k * (k + 1) / (n - k - 1)
    hqic = 2 * k * np.log(np.log(n)) - 2 * log_likelihood
    return aic, caic, hqic


def format_results_with_criteria(results, data):
    print("MLE estimates, SEs, and information criteria for the dataset.")
    print(f"{'Model':<15}{'Log_L':<15}{'AIC':<15}{'CAIC':<15}{'HQIC':<15}")
    n = len(data)  # 数据点数量
    for dist_name, result in results.items():
        params = result['params']
        ses = result['se']
        # 计算对数似然值
        log_likelihood_func = distributions[dist_name]
        log_likelihood = -log_likelihood_func(params, data)  # 负对数似然

        # 计算AIC、CAIC、HQIC
        k = len(params)  # 参数个数
        aic, caic, hqic = calculate_info_criteria(log_likelihood, k, n)

        if dist_name == "UEXE":
            print(f"{dist_name}(λ,δ)".ljust(15) +
                  f"{log_likelihood:.6f}".ljust(15) +
                  f"{aic:.6f}".ljust(15) +
                  f"{caic:.6f}".ljust(15) +
                  f"{hqic:.6f}".ljust(15))
        elif dist_name == "Beta":
            print(f"{dist_name}(κ,γ)".ljust(15) +
                  f"{log_likelihood:.6f}".ljust(15) +
                  f"{aic:.6f}".ljust(15) +
                  f"{caic:.6f}".ljust(15) +
                  f"{hqic:.6f}".ljust(15))
        elif dist_name == "Kumaraswamy":
            print(f"Kum(a,b)".ljust(15) +
                  f"{log_likelihood:.6f}".ljust(15) +
                  f"{aic:.6f}".ljust(15) +
                  f"{caic:.6f}".ljust(15) +
                  f"{hqic:.6f}".ljust(15))
        elif dist_name == "Unit Gamma":
            print(f"UG(θ,ρ)".ljust(15) +
                  f"{log_likelihood:.6f}".ljust(15) +
                  f"{aic:.6f}".ljust(15) +
                  f"{caic:.6f}".ljust(15) +
                  f"{hqic:.6f}".ljust(15))
        elif dist_name == "Power Topp-Leone":
            print(f"PTL(ν,τ)".ljust(15) +
                  f"{log_likelihood:.6f}".ljust(15) +
                  f"{aic:.6f}".ljust(15) +
                  f"{caic:.6f}".ljust(15) +
                  f"{hqic:.6f}".ljust(15))
        elif dist_name == "Bounded Weighted Exponential":
            print(f"BWE(σ,η)".ljust(15) +
                  f"{log_likelihood:.6f}".ljust(15) +
                  f"{aic:.6f}".ljust(15) +
                  f"{caic:.6f}".ljust(15) +
                  f"{hqic:.6f}".ljust(15))
        elif dist_name == "Unit Burr XII":
            print(f"UBXII(μ,ϕ)".ljust(15) +
                  f"{log_likelihood:.6f}".ljust(15) +
                  f"{aic:.6f}".ljust(15) +
                  f"{caic:.6f}".ljust(15) +
                  f"{hqic:.6f}".ljust(15))
        elif dist_name == "Unit Birnbaum-Saunders":
            print(f"UBS(α,β)".ljust(15) +
                  f"{log_likelihood:.6f}".ljust(15) +
                  f"{aic:.6f}".ljust(15) +
                  f"{caic:.6f}".ljust(15) +
                  f"{hqic:.6f}".ljust(15))
        elif dist_name == "Unit Inverse Gaussian":
            print(f"UIG(ζ,ω)".ljust(15) +
                  f"{log_likelihood:.6f}".ljust(15) +
                  f"{aic:.6f}".ljust(15) +
                  f"{caic:.6f}".ljust(15) +
                  f"{hqic:.6f}".ljust(15))
        elif dist_name == "Unit Mirra":
            print(f"UM(ψ,Θ)".ljust(15) +
                  f"{log_likelihood:.6f}".ljust(15) +
                  f"{aic:.6f}".ljust(15) +
                  f"{caic:.6f}".ljust(15) +
                  f"{hqic:.6f}".ljust(15))


# 打印数据集1结果，包含信息准则
format_results_with_criteria(results_data1, data1)
print()
# 打印数据集2结果，包含信息准则
format_results_with_criteria(results_data2, data2)


# 计算Kolmogorov-Smirnov检验的统计量和p值
def kolmogorov_smirnov(data, cdf_func, params):
    n = len(data)
    sorted_data = np.sort(data)
    cdf_vals = cdf_func(sorted_data, *params)
    ks_stat = np.max(np.abs((np.arange(1, n + 1) / n) - cdf_vals))
    # p值近似计算
    p_value = np.exp(-2 * n * ks_stat ** 2)
    return ks_stat, p_value


# 计算Anderson-Darling检验的统计量
def anderson_darling(data, cdf_func, params):
    sorted_data = np.sort(data)
    n = len(sorted_data)
    cdf_vals = cdf_func(sorted_data, *params)
    ad_stat = -n - np.sum((2 * np.arange(1, n + 1) - 1) * np.log(cdf_vals) +
                          (2 * (n - np.arange(1, n + 1)) - 1) * np.log(1 - cdf_vals))
    return ad_stat


# 计算Cramer-VonMises检验的统计量
def cramer_von_mises(data, cdf_func, params):
    sorted_data = np.sort(data)
    n = len(sorted_data)
    cdf_vals = cdf_func(sorted_data, *params)
    w_stat = (1 / n) * np.sum((cdf_vals - (2 * np.arange(1, n + 1) - 1) / n) ** 2)
    return w_stat


# 定义分布的累计分布函数 (CDF)，需要根据每个分布调整
def uexe_cdf(data, delta, lambd):
    return 1 - np.exp(-lambd * np.log(data) ** delta)


def beta_cdf(data, a, b):
    # 使用自定义CDF实现
    return np.cumsum(data ** (a - 1) * (1 - data) ** (b - 1)) / np.sum(data ** (a - 1) * (1 - data) ** (b - 1))


def kumaraswamy_cdf(data, a, b):
    return 1 - (1 - data ** a) ** b


def unit_gamma_cdf(data, a, b):
    return 1 - np.exp(-b * np.log(data) ** a)


def power_topp_leone_cdf(data, a, b):
    return 1 - (1 - data ** a) ** b


def bounded_weighted_exponential_cdf(data, a, b):
    return 1 - (1 - data ** a) ** b


def unit_burr_xii_cdf(data, a, b):
    return 1 - np.exp(-(np.log(data) ** b) ** a)


def unit_birnbaum_saunders_cdf(data, a, b):
    return 1 - np.exp(-a * (data ** 2) / (b ** 2))


def unit_inverse_gaussian_cdf(data, a, b):
    return 1 - np.exp(-((np.log(data) + b) ** 2) / (2 * a))


def unit_mirra_cdf(data, a, b):
    return 1 - np.exp(-b * (data ** a))


# 计算分布的MLE并进行K-S, A*和W*检验
def fit_and_test_distribution(distributions, initial_guesses, data):
    results = []
    for dist_name, log_likelihood_func in distributions.items():
        # 估计参数
        initial_guess = initial_guesses[dist_name]
        params, _ = mle_estimation(log_likelihood_func, initial_guess, data)

        # 选择对应的CDF函数
        cdf_func = globals()[f'{dist_name.lower().replace(" ", "_")}_cdf']

        # 计算统计量
        ks_stat, ks_p_value = kolmogorov_smirnov(data, cdf_func, params)
        ad_stat = anderson_darling(data, cdf_func, params)
        w_stat = cramer_von_mises(data, cdf_func, params)

        results.append({
            'Model': dist_name,
            'A*': ad_stat,
            'W*': w_stat,
            'K-S': ks_stat,
            'p-value(K-S)': ks_p_value
        })

    return results


# 执行所有分布的检验
results = fit_and_test_distribution(distributions, initial_guesses, data1)

# 输出结果
for result in results:
    print(f"{result['Model']} {result['A*']:.6f} {result['W*']:.6f} {result['K-S']:.6f} {result['p-value(K-S)']:.6f}")


