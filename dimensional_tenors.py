"""
Higher dimensional tensors
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

one_d = torch.arange(0, 9)
two_d = one_d.view(3, 3)
two_d.dim()
# This returns 2

#Slicing 3D tensors
x = torch.arange(0, 18).view(2, 3, 3)
x_13 = x[1, 1, 1]
x_13 = x[1][1][1]
#Ranges in slices
stuff = x[1, 0:2, 0:3]
stuff2 = x[1, :, :]