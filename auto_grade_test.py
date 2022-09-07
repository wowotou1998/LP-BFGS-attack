import torch

x = torch.ones(3, 4).requires_grad_(True)*2
# for i in range(3):
#     for j in range(4):
#         x[i][j] = i + j
y = x ** 2
print(x)
print(y)
weight = torch.ones(y.size())
print(weight)
dydx = torch.autograd.grad(outputs=y,
                           inputs=x,
                           grad_outputs=weight,
                           retain_graph=True,
                           create_graph=True,
                           only_inputs=True)
"""(x**2)' = 2*x """
print(dydx)
d2ydx2 = torch.autograd.grad(outputs=dydx[0],
                             inputs=x,
                             grad_outputs=weight,
                             retain_graph=True,
                             create_graph=True,
                             only_inputs=True)
print(d2ydx2[0])