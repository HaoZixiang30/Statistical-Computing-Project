'''
@File: config.py
Desc: Parameters for Experiments
'''
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2024, help='Seed')
    parser.add_argument('--K', type=int, default=1000, help='Number of samples')
    parser.add_argument('--n_list', type=list, default=[100, 150, 200, 300, 400, 500], help='Number of items in each sample')
    return parser.parse_args()