import torch

a = torch.tensor([[[1,2]]])
print(a.flatten())
print(a.view(-1))
