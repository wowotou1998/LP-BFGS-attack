import pickle

with open('./Checkpoint/%s' % ('data_2022_10_13_19_53_51.pkl'), 'rb') as f:
    data = pickle.load(f)
    for i in data:
        print(i)