from pickletools import optimize
import torch 

x = torch.randn(3, requires_grad=True) #add it to the computational graph and shows what operation to do to go back for backward propgration 
print(x)

y = x+2
print(y)
z = y*y*2
print (z)
z = z.mean()
print(z)

z.backward()
print(x.grad) #has gradients

#how to prevent tracking gradients
#prevents tracking history 
# x.requires_grad_(False)
#x.detach()
#with torch.no_grad():

weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_() #must clear gradients so weights are correct

#does same step to clear gradients 
optimizer = torch.optim.SGD(weights, lr=0.01)
optimizer.step()
optimizer.zero_grad()