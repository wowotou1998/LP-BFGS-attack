import torch

a = torch.tensor([[[1,2]]])
print(a.flatten())
print(a.view(-1))
b = a.flatten()
v,idx = b.sort(descending=True)
b[idx] =3
print(v,idx)
print(b)
