"""
Matrix multiplication
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

mat_a = torch.tensor((0, 3, 5, 5, 5, 2)).view(2, 3)
mat_b = torch.tensor((3, 4, 3, -2, 4, -2)).view(3, 2)
multiplied_matrix = torch.matmul(mat_a, mat_b)
# The same thing happens when you do 'mat_a @ mat_b'


# Now for derivatives
# Attempt at computing gradient would throw an error if 
# this field is not True
x = torch.tensor(2.0, requires_grad=True)
y = 9*x**4 + 2*x**3 + 3*x**2 + 6*x + 1
# Actually does the differentiation
y.backward()
# Gives the value at x
x.grad

# Partial derivatives
x = torch.tensor(1.0, requires_grad=True)
z = torch.tensor(2.0, requires_grad=True)
y = x**2 + z**3
y.backward()
x.grad
z.grad