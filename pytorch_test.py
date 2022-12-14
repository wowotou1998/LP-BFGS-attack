import torch

a = torch.tensor([[[1, 2]]])
print(torch.cuda.is_available())
print('a[...]', a[...][0][0])
# 取行
# print(a[[0,1],:])
# 取列
print(a[:, :, [1]])

print(a.flatten())
print(a.view(-1))
b = a.flatten()
v, idx = b.sort(descending=True)
b[idx] = 3
print(v, idx)
print(b)

c = torch.tensor([1, 2])
d = torch.tensor([2, 2])
print(c / d)
print(c)
