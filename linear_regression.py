"""
Linear Regression
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Linear
import torch.nn as nn

##Creating the original weight and bias values, requires
##gradient descent functions to occur
#w = torch.tensor(3.0, requires_grad=True)
#b = torch.tensor(1.0, requires_grad=True)
#
##Just defining the function
#def forward(x):
#    y = w*x + b
#    return y
#
##Constructing the values that will be passed into the function
#x = torch.tensor([[4], [7]])
#print(forward(x))

"""
Linear Regression part 2
"""

##Just keeping random values consistent
#torch.manual_seed(1)
#
##For every input there is a single output
##This creates a linear model
#model = Linear(in_features=1, out_features=1)
#print(model.bias, model.weight)
#
#x = torch.tensor([[2.0], [3.3]])
#print(model(x))

"""
Linear Regression part 3
"""
#
#class LR(nn.Module):
#    
#    def __init__(self, input_size, output_size):
#        super().__init__()
#        self.linear = nn.Linear(input_size, output_size)
#    
#    def forward(self, x):
#        pred = self.linear(x)
#        return pred
#        
#torch.manual_seed(1)
#model = LR(1, 1)
#x = torch.tensor([[1.0], [2.0]])
#print(model.forward(x))        
#        
"""
Linear Regression part 4
"""
#

#class LR(nn.Module):
#    
#    def __init__(self, input_size, output_size):
#        super().__init__()
#        self.linear = nn.Linear(input_size, output_size)
#        
#    def forward(self, x):
#        pred = self.linear(x)
#        return pred
#
#
#torch.manual_seed(1)
#model = LR(1, 1)
#w, b = model.parameters()
#
#
#def get_params():
#    return (w[0][0].item(), b[0].item())
#
#def plot_fit(title):
#    plt.title = title
#    w1, b1 = get_params()
#    x1 = np.array((-30, 30))
#    y1 = w1*x1 + b1
#    plt.plot(x1, y1, 'r')
#    plt.scatter(x, y)
#    plt.show()
#
#x = torch.randn(100, 1) * 10
#y = x + torch.randn(100, 1) * 3
#plt.plot(x.numpy(), y.numpy(), 'o')
#plt.ylabel('y')
#plt.xlabel('x')
#plot_fit("Initial Model")

"""
Linear Regression part 5
"""

class LR(nn.Module):
    
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        pred = self.linear(x)
        return pred
    
torch.manual_seed(1)
model = LR(1, 1)
w, b = model.parameters()

def get_params():
    return (w[0][0].item(), b[0].item())

def plot_fit(title):
    plt.title = title
    w1, b1 = get_params()
    x1 = np.array((-30, 30))
    y1 = w1*x1 + b1
    plt.plot(x1, y1, 'r')
    plt.scatter(x, y)
    plt.show()
    
x = torch.randn(100, 1) * 10
y = x + torch.randn(100, 1) * 3
plt.plot(x.numpy(), y.numpy(), 'o')
plt.ylabel('y')
plt.xlabel('x')
plot_fit("Initial Model")


criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=.01)
epochs = 100
losses = []

for i in range(epochs):
    y_pred = model.forward(x)
    loss = criterion(y_pred, y)
    print("epoch: ", i, "loss: ", loss.item())
    
    losses.append(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
