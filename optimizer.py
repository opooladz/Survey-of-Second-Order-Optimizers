import torch
from torch.optim.optimizer import Optimizer

device = torch.device('cuda:0')

class LM(Optimizer):
    '''
    Arguments:
        lr: learning rate (step size)
        alpha: the hyperparameter in the regularization
        prev_dw: keep track of the previous delta_w as a momentum term 
        eps, dP: two hyperparameter in the CG momentum

    '''
    def __init__(self, params, lr=1, alpha=1, eps=0.9, dP=0.6):
        defaults = dict(
            lr = lr,
            alpha = alpha,
            prev_dw = None,
            eps = eps,
            dP = dP
        )
        super(LM, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError ("LM doesn't support per-parameter options") 

        self._params = self.param_groups[0]['params']

    def step(self, closure=None):
        '''
        performs a single step
        in the closure: we approximate the Hessian for cross entropy loss

        '''
        assert len(self.param_groups) == 1
        group = self.param_groups[0]
        lr = group['lr']
        alpha = group['alpha']
        eps = group['eps']
        dP = group['dP']
        prev_dw = group['prev_dw']

        prev_loss, g, H = closure(sample=True)
        record_loss = prev_loss.item()
        print ('prev loss:{}'.format(prev_loss.item()))
        
        H += torch.eye(H.shape[0]).to(device)*alpha
        H_inv = torch.inverse(H)

        if prev_dw is None:
            delta_w = (-H_inv @ g).detach() 
        else:
            I_GG = torch.squeeze(g.T @ H_inv @ g)
            I_FF = torch.squeeze(prev_dw.T @ H @ prev_dw)
            I_GF = torch.squeeze(g.T @ prev_dw)
            dQ = -eps * dP * torch.sqrt(I_GG)
            t2 = 0.5 / torch.sqrt((I_GG * (dP**2) - dQ**2) / (I_FF*I_GG - I_GF*I_GF))
            t1 = (-2*t2*dQ + I_GF) / I_GG
            print ('t1:{}'.format(t1))
            print ('t2:{}'.format(t2))
            delta_w = (-t1/t2 * H_inv @ g + 0.5/t2 * prev_dw).detach()
        
        offset = 0
        for p in self._params:
            numel = p.numel()
            with torch.no_grad():
                p.add_(delta_w[offset:offset + numel].view_as(p),alpha=lr)
            offset += numel

        outputs, loss = closure(sample=False)
        print ('loss:{}'.format(loss.item()))
        group['prev_dw'] = delta_w

        if loss < prev_loss:
            print ('successful iteration')
            if alpha >= 1e-5:
                group['alpha'] /= 10
        else:
            print ('failed iteration')
            if alpha <= 1e5:
                group['alpha'] *= 100
            # undo the step
            offset = 0
            for p in self._params:    
                numel = p.numel()
                with torch.no_grad():
                    p.sub_(delta_w[offset:offset + numel].view_as(p),alpha=lr)
                offset += numel

        return outputs, record_loss
