import pickle

with open('./Checkpoint/%s' % ('data.pkl'), 'rb') as f:
    data = pickle.load(f)
    print(data)