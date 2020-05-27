import torch.nn as nn
import torch
import numpy as np
from torch.optim.optimizer import Optimizer
import functools
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torch.autograd import Variable
import torch_dct as dct
import numpytorch
from timeit import default_timer as timer
import scipy.io
import sys
import traceback
# import fast_inv



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
class LM(Optimizer):
    '''
    Arguments:
        lr: learning rate (step size) default:1
        alpha: the hyperparameter in the regularization default:0.2
    '''
    def __init__(self, params, blocks=10, lr=1, alpha=1e-2,save_hessian=False):
        defaults = dict(
            lr = lr,
            alpha = alpha,
            blocks = blocks,
            save_hessian = save_hessian
        )
        super(LM, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError ("LM doesn't support per-parameter options")

        self._params = self.param_groups[0]['params']

    def step(self, closure=None):
        '''
        performs a single step
        in the closure: we evaluate the diff

        '''
        assert len(self.param_groups) == 1
        group = self.param_groups[0]
        lr = group['lr']
        alpha = group['alpha']
        params = group['params']
        blocks = group['blocks']

        diff= closure()

        # calculate Jacobian
        J = jacobian(diff,params , create_graph=True, retain_graph=True).detach()
        prev_loss = torch.mean(diff.detach() ** 2)
        # approximate Hessian
        H = torch.matmul(J.T, J) + torch.eye(J.shape[-1]).to('cuda:0') * alpha   
        with torch.no_grad():
            # H_inv = fast_inv.inv_block_gpu(H,blocks)
            H_inv = torch.inverse(H)
        # calculate the update

        delta_w = -1 * torch.matmul(H_inv, torch.matmul(J.T, diff)).detach()
        # print(delta_w.shape)
        offset = 0
        for p in group['params']:
            numel = p.numel()
            with torch.no_grad():
                p.add_(delta_w[offset:offset + numel].view_as(p),alpha=lr)
            offset += numel

        diff = closure()
        loss = torch.mean(diff.detach() ** 2)
        # print (loss.item())
        if loss < prev_loss:
            # print('Succssful Iteration Loss: {}'.format(loss.item()))
            # print ('successful iteration')
            if alpha > 1e-10:
                group['alpha'] /= 10
            return loss.item()
        else:
            # print ('failed iteration')
            if alpha < 1e10:
                group['alpha'] *= 10
            # undo the step
            offset = 0
            for p in group['params']:
                numel = p.numel()
                with torch.no_grad():
                    p.sub_( delta_w[offset:offset + numel].view_as(p),alpha=lr)
                offset += numel
