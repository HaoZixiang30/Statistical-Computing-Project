'''
@File: main.py
Desc: Main for simulation
'''
from DataGenerator import DataGenerator
from MLE_Estimate import MLE_Estimate
from config import parse_args
import numpy as np
import pickle
import logging
import os
logging.basicConfig(level=logging.INFO)

def data_generation(opt):
    K = opt.K
    n_list = opt.n_list
    logging.info("-------------Generating Data-------------")
    datagenerator = DataGenerator(opt)
    data1 = datagenerator(K, n_list, delta=0.5, lambdaa=0.5)
    data2 = datagenerator(K, n_list, delta=1, lambdaa=2)
    data3 = datagenerator(K, n_list, delta=3, lambdaa=4)
    logging.info("-------------Saving Data-------------")
    pickle.dump(data1, open('SimulationData_Case1.pkl', 'wb'))
    pickle.dump(data2, open('SimulationData_Case2.pkl', 'wb'))
    pickle.dump(data3, open('SimulationData_Case3.pkl', 'wb'))
    logging.info("-------------Data Saved-------------")
    return data1, data2, data3

def mle_estimate(opt, data):
    '''
    @Params:
    opt: configuration
    data: dic{K:{n:[sample list]}}, where length of [sample list] is n

    @Returns:
    ls_kmean: dic{n:[mean(delta list), mean(lambda list)]}, where n:[100, 150, 200, 300, 400, 500]
    ls_k: dic{n:[[delta list], [lambda list]]}, where n:[100, 150, 200, 300, 400, 500]
    '''
    mle = MLE_Estimate(opt)
    ls_kmean = {}
    ls_k = {}
    for n in opt.n_list:
        ls_d = []
        ls_l = []
        for k in data:
            data_ls = np.array(data[k][n])
            delta, lambdaa = mle(data_ls)
            ls_d.append(delta)
            ls_l.append(lambdaa)
        if n not in ls_kmean:
            ls_kmean[n] = [np.mean(ls_d), np.mean(ls_l)]
            ls_k[n] = [ls_d, ls_l]

    return ls_kmean, ls_k
   
        

def main():
    opt = parse_args()

    if os.path.exists('SimulationData_Case1.pkl') \
    and os.path.exists('SimulationData_Case2.pkl') \
    and os.path.exists('SimulationData_Case3.pkl'):
        data1 = pickle.load(open('SimulationData_Case1.pkl', 'rb'))
        data2 = pickle.load(open('SimulationData_Case2.pkl', 'rb'))
        data3 = pickle.load(open('SimulationData_Case3.pkl', 'rb'))
    else:
        data1, data2, data3 = data_generation(opt)
    case1_mean, case1_ls = mle_estimate(opt, data1)
    opt.delta_init = 1
    opt.lambda_init = 2
    case2_mean, case2_ls = mle_estimate(opt, data2)
    opt.delta_init = 3
    opt.lambda_init = 4
    case3_mean, case3_ls = mle_estimate(opt, data3)


if __name__ == '__main__':
    main()