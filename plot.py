import pickle
import time

start = time.perf_counter()
with open('./Checkpoint/%s' % ('data_2022_10_13_19_53_51.pkl'), 'rb') as f:
    data = pickle.load(f)
    for i in data:
        print(i)
end = time.perf_counter()
print(end - start)
a = [2]*6
print(a)