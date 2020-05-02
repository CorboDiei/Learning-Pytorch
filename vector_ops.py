"""
Vector Operations
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
t_one = torch.tensor((1, 2, 3))
t_two = torch.tensor((1, 2, 3))
# Vector addition just using '+'
# Vector multiplication by element just using '*'
dot_product = torch.dot(t_one, t_two)
one_to_ten = torch.linspace(0, 10, 5)
# Makes a tensor that essentially has the same function as 
# range, with a beginning, end, and optionally a step
# The default step is 100

# We can make all of the x values this linspace with
x = torch.linspace(0, 10, 5)
y = torch.exp(x)

# plt works with numpy data structures and not pytorch currently

new_x = x.numpy()
new_y = y.numpy()

plt.plot(new_x, new_y)
