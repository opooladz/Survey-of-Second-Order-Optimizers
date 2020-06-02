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
import copy
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
device = torch.device('cuda:0')



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
    def __init__(self, params, lr=1, alpha=1,eps=0.9, dP=0.6):
        defaults = dict(
            lr = lr,
            alpha = alpha,
            prev_dw = None,
            prev_dw1 = None,
            eps = eps,
            dP = dP
            )
        super(LM, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError ("LM doesn't support per-parameter options")

        self._params = self.param_groups[0]['params']

    def _gather_flat_grad(self):
        views = list()
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.contiguous().view(-1)
            views.append(view)
        return torch.cat(views, 0)
        
    def step(self, closure=None,dg_prev=None,cos=None,lr_linesearch=False):
        '''
        performs a single step
        in the closure: we evaluate the diff

        '''
        assert len(self.param_groups) == 1
        group = self.param_groups[0]
        lr = group['lr']
        alpha = group['alpha']
        params = group['params']
        # blocks = group['blocks']
        eps = group['eps']
        dP = group['dP']
        prev_dw = group['prev_dw']
        prev_dw1 = group['prev_dw']        

        diff = closure(sample=True)

        # calculate Jacobian
        J = jacobian(diff,params , create_graph=True, retain_graph=True).detach()
        g = torch.matmul(J.T, diff)
        prev_loss = torch.mean(diff.detach() ** 2)
        # approximate Hessian
        H = torch.matmul(J.T, J) 
        dg = torch.diag(H)
        dg = torch.max(dg_prev,dg)
        H += torch.diag(dg).to(device)*alpha         
        with torch.no_grad():
            H_inv = torch.inverse(H)

        # CG Momentum + calculate the update
        if prev_dw is None:
            delta_w = delta_w1 = (-H_inv @ g).detach() 
            group['prev_dw1'] = delta_w1
        else:
            I_GG = torch.squeeze(g.T @ H_inv @ g)
            I_FF = torch.squeeze(prev_dw.T @ H @ prev_dw)
            I_GF = torch.squeeze(g.T @ prev_dw)
            dQ = -eps * dP * torch.sqrt(I_GG)
            t2 = 0.5 / torch.sqrt((I_GG * (dP**2) - dQ**2) / (I_FF*I_GG - I_GF*I_GF))
            t1 = (-2*t2*dQ + I_GF) / I_GG
            print ('t1:{}'.format(t1))
            print ('t2:{}'.format(t2))
            delta_w1 = -1*H_inv @ g
            group['prev_dw1'] = delta_w1
            delta_w = (t1/t2 *delta_w1  + 0.5/t2 * prev_dw).detach()
            del I_GG
            del I_FF
            del dP
            del dQ
        del H
        del H_inv       

        offset = 0
        for p in group['params']:
            numel = p.numel()
            with torch.no_grad():
                p.add_(delta_w[offset:offset + numel].view_as(p),alpha=lr)
            offset += numel
        outputs, diff = closure(sample=False)
        loss = torch.mean(diff.detach() ** 2) 
        print ('loss:{}'.format(loss.item()))
        group['prev_dw'] = delta_w


        if loss < prev_loss:
            print ('successful iteration')
            if (prev_loss - loss)/loss > 0.75 and alpha >= 1e-5:
                group['alpha'] *=2/3 
            return outputs, loss, dg 
        elif prev_dw is not None:
            beta = cos(delta_w1,prev_dw)
            b = 2
            tmp = (1-beta)**b *loss
            print((1-beta)**b)
            print(tmp)
            print(loss)
            print(prev_loss)
            if tmp <=prev_loss:
                print('Accepting Uphill Step')
                if (prev_loss - loss)/loss > 0.75 and alpha >= 1e-5:
                    # not sure if this is the best way to go maybe dont change the alpha 
                    group['alpha'] *= 2/3
                return outputs, loss, dg  
            else:
                print ('failed iteration')
                # undo the step
                offset = 0
                for p in self._params:    
                    numel = p.numel()
                    with torch.no_grad():
                        p.sub_(delta_w[offset:offset + numel].view_as(p),alpha=lr)
                    offset += numel                           
                if alpha <= 1e5 and (prev_loss - loss)/loss < 0.25 :
                    group['alpha'] *= 3/2        
                dg = dg_prev
        return outputs, prev_loss, dg

