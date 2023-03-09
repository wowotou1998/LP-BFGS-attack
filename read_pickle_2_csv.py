import pickle

with open('./Checkpoint/ce_iter200_1_100acc_20pixel_1e3_CIFAR10_2023_03_08_16_32_00.pkl', 'rb') as f:
    csv_data = pickle.load(f)
import pandas as pd

job_name = 'ce_iter200_%d_100acc_20pixel_1e3' % 1000
csv = pd.DataFrame(columns=csv_data[0], data=csv_data[1:])
csv.to_csv('./Checkpoint/%s_%s_%s.csv' % ('ce_iter200_1_100acc_20pixel_1e3_CIFAR10_2023_03_08_16_32_00', 'dataset_i', 'current_time'))
