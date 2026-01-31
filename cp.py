import numpy as np
import scipy.stats as stats
import pickle
from main import main
from config import parse_args
from DataGenerator import DataGenerator
from MLE_Estimate import MLE_Estimate
from config import parse_args
import numpy as np
import pickle
import logging
import os
logging.basicConfig(level=logging.INFO)


file_path = 'SimulationData_Case3.pkl'
with open(file_path, 'rb') as file:
    sample = pickle.load(file)# sample 的形式{}

opt = parse_args()
# def data_generation(opt):
#     K = opt.K
#     n_list = opt.n_list
#     logging.info("-------------Generating Data-------------")
#     datagenerator = DataGenerator(opt)
#     data1 = datagenerator(K, n_list, delta=0.5, lambdaa=0.5)
#     data2 = datagenerator(K, n_list, delta=1, lambdaa=2)
#     data3 = datagenerator(K, n_list, delta=3, lambdaa=4)
#     logging.info("-------------Saving Data-------------")
#     pickle.dump(data1, open('SimulationData_Case1.pkl', 'wb'))
#     pickle.dump(data2, open('SimulationData_Case2.pkl', 'wb'))
#     pickle.dump(data3, open('SimulationData_Case3.pkl', 'wb'))
#     logging.info("-------------Data Saved-------------")
#     return data1, data2, data3
# data1, data2, data3 = data_generation(opt)
# print(data2[1][700])


# 假设给定的对数似然函数的二阶导数，构造信息矩阵的元素
def second_derivatives(params, x):
    delta, lam = params
    n = len(x)
    x = np.array(x)
    
    # 计算对数似然函数的二阶偏导数（示例计算）
    d2ldelta2 = (2 * n / delta**2) - (n / (delta + lam)**2)
    d2ldelta_lambda = -n / (delta + lam)**2  # 示例表达式
    d2lambd2 = -n / (delta + lam)**2 + np.sum(np.log(x)**2 / (1 - lam * np.log(x))**2)  # 示例表达式
    
    # 返回信息矩阵的元素
    return d2ldelta2, d2lambd2, d2ldelta_lambda

# 计算信息矩阵和其逆矩阵
def compute_information_matrix(params, x):
    d2ldelta2, d2lambd2, d2ldelta_lambda = second_derivatives(params, x)
    
    # 构造信息矩阵
    info_matrix = np.array([[d2ldelta2, d2ldelta_lambda],
                            [d2ldelta_lambda, d2lambd2]])
    
    # 计算信息矩阵的逆
    def inverse_2x2(matrix):
        # 提取元素
        a, b = matrix[0, 0], matrix[0, 1]
        c, d = matrix[1, 0], matrix[1, 1]
        
        # 计算行列式
        determinant = a * d - b * c
        if determinant == 0:
            raise ValueError("矩阵不可逆")
        
        # 计算逆矩阵
        inverse = (1 / determinant) * np.array([[d, -b],
                                                [-c, a]])
        return inverse


    # info_matrix_inv = np.linalg.inv(info_matrix)
    info_matrix_inv = inverse_2x2(info_matrix)
    
    return info_matrix_inv

def calculate_intertval_case1():
    case1_params, case2_params, case3_params= main()
    n_size = [600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
    intervals = {}
    
    for size in n_size:
        delta_list, lam_list = case3_params[size]
        intervals[size] = {}
        i=0
        for j in range(len(delta_list)):
            

            params = [delta_list[j], lam_list[j]]
            data = sample[j][size]
            I_inv = compute_information_matrix(params, data)
            # 获取方差
            var_delta = I_inv[0, 0]  # delta的方差
            var_lambda = I_inv[1, 1]  # lambda的方差

            # if var_lambda < 0 :
            #     continue


            # 计算标准误差
            se_delta = np.sqrt(abs(var_delta))
            se_lambda = np.sqrt(abs(var_lambda))

            # 计算95%置信区间
            Z_critical = stats.norm.ppf(0.975)  # Z值，对于95%置信区间，

            # 置信区间
            conf_interval_delta = (delta_list[j] - Z_critical * se_delta, delta_list[j] + Z_critical * se_delta)
            conf_interval_lambda = (lam_list[j] - Z_critical * se_lambda, lam_list[j] + Z_critical * se_lambda)
            intervals[size][i] = [conf_interval_delta, conf_interval_lambda]
            i += 1

    return intervals

interval = calculate_intertval_case1()



for j in [600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]:
    l_delta = 0
    r_delta = 0
    l_lam, r_lam =0 ,0
    delta_len, lam_len = 0,0
    for k,v in interval[j].items():
        l_delta += v[0][0] 
        r_delta += v[0][1]
        l_lam  += v[1][0]
        r_lam += v[1][1]
        delta_len += v[0][1] - v[0][0] 
        lam_len += v[1][1]-v[1][0]
    # 计算覆盖率

    print(j ,"AL:",delta_len/len(interval[j]), lam_len/len(interval[j]))

def calculate_coverage(true_delta, true_lambda):
    coverage = {}
    
    n_size = [600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
    for size in n_size:
        coverage_count_delta = 0
        coverage_count_lambda = 0
        interval = calculate_intertval_case1()[size]
        coverage[size] = {}
        
    
        for i in range(len(interval)):
            conf_interval_delta, conf_interval_lam = interval[i]
            # 检查置信区间是否覆盖真实的delta和lambda
            if conf_interval_delta[0] <= true_delta and true_delta <= conf_interval_delta[1]:
                coverage_count_delta += 1
            
            if conf_interval_lam[0] <= true_lambda and true_lambda <= conf_interval_lam[1]:  # 示例的lambda覆盖
                coverage_count_lambda += 1
    
        # 计算覆盖率
        delta_coverage_rate = coverage_count_delta / len(interval)
        lambda_coverage_rate = coverage_count_lambda / len(interval)
        coverage[size]['delta'] = delta_coverage_rate
        coverage[size]['lam'] = lambda_coverage_rate
    
    return coverage

coverage = calculate_coverage(true_delta=3.0, true_lambda=4.0)

print(coverage)