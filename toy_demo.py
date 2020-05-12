#%%
import torch
import torch.nn as nn
import torch
import numpy as np
from torch.optim.optimizer import Optimizer
#%%
device = torch.device('cuda')
#%%
def gradient(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False):
    '''
    Compute the gradient of `outputs` with respect to `inputs`

    gradient(x.sum(), x)
    gradient((x * y).sum(), [x, y])
    '''
    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)
    grads = torch.autograd.grad(outputs, inputs, grad_outputs,
                                allow_unused=True,
                                retain_graph=retain_graph,
                                create_graph=create_graph)
    grads = [x if x is not None else torch.zeros_like(y) for x, y in zip(grads, inputs)]
    return torch.cat([x.contiguous().view(-1) for x in grads])


def jacobian(outputs, inputs, create_graph=False,retain_graph=True):
    '''
    Compute the Jacobian of `outputs` with respect to `inputs`

    jacobian(x, x)
    jacobian(x * y, [x, y])
    jacobian([x * y, x.sqrt()], [x, y])
    '''
    if torch.is_tensor(outputs):
        outputs = [outputs]
    else:
        outputs = list(outputs)

    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)

    jac = []
    for output in outputs:
        output_flat = output.view(-1)
        output_grad = torch.zeros_like(output_flat)
        for i in range(len(output_flat)):
            output_grad[i] = 1
            jac += [gradient(output_flat, inputs, output_grad, retain_graph, create_graph)]
            output_grad[i] = 0
    return torch.stack(jac)
#%%
inputs = torch.randn(10, requires_grad=True)
net = nn.Linear(10, 3)
outputs = net(inputs)
outputs_flat = outputs.view(-1)
outputs_grad = torch.zeros_like(outputs_flat)
jac = []
for i in range(len(outputs_flat)):
    outputs_grad[i] = 1
    jac += [gradient(outputs_flat, inputs, outputs_grad, retain_graph=True, create_graph=False)]
    outputs_grad[i] = 0

torch.stack(jac).shape
#%%
class LM(Optimizer):
    '''
    Arguments:
        lr: learning rate (step size) default:1
        alpha: the hyperparameter in the regularization default:0.2
    '''
    def __init__(self, params, lr=1):
        defaults = dict(
            lr = lr
            #alpha = alpha
        )
        super(LM, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError ("LM doesn't support per-parameter options")    
    
    def step(self,net, f, J, alpha, closure=None):
        '''
        performs a single step

        '''
        assert len(self.param_groups) == 1
        group = self.param_groups[0]
        lr = group['lr']
        # approximate Hessian
        H = torch.matmul(J.T, J) + torch.eye(J.shape[-1]).to(device) * alpha
        # calculate the update       
        delta_w = -1 * torch.matmul(torch.inverse(H), torch.matmul(J.T, f)).detach()
        offset = 0
        for p in group['params']:
            numel = p.numel()
            p = p + lr * delta_w[offset:offset + numel].view_as(p)
            offset += numel
        net()
        
        
#%%
net = torch.nn.Sequential(
        torch.nn.Linear(1, 50),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(50, 20),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(20, 1),
    )
net.cuda('cuda:0')

optimizer = LM(net.parameters())
BATCH_SIZE = 64
EPOCH = 200
x = torch.unsqueeze(torch.linspace(-10, 10, 1000), dim=1)  
y = torch.sin(x) + 0.2*torch.rand(x.size())
torch_dataset = torch.utils.data.TensorDataset(x, y)

loader = torch.utils.data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True)
#%%
prev_loss = float('inf')
alpha = 0.5
for i, (data, target) in enumerate(torch_dataset):
    print (i)
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    out = net(data)
    diff = out - target
    loss = (diff ** 2).item()
    print ('MSE loss')
    print (loss)
    J = jacobian(diff, net.parameters())
    optimizer.step(net,target, J, alpha)
    if loss < prev_loss:
        print ('successful iteration')
        if alpha > 1e-5:
            alpha /= 10
    else:
        if alpha < 1e5:
            alpha *= 10
    prev_loss = loss

