import torch

x = torch.tensor(1.0)
x._requires_grad = True

y1 = torch.tensor(2.0)
y2 = torch.tensor(3.0)

l1 = x * y1
l2 = x * y2

l1.backward()
l2.backward()
