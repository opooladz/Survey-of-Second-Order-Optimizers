import torch
from torch.optim.optimizer import Optimizer

device = torch.device('cuda:0')

class LM(Optimizer):
    '''
    Arguments:
        lr: learning rate (step size)
        alpha: the hyperparameter in the regularization
    '''
    def __init__(self, params, lr=0.5, alpha=10):
        defaults = dict(
            lr = lr,
            alpha = alpha
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
        params = group['params']

        prev_loss, g, H = closure(sample=True)
        record_loss = prev_loss.item()
        print ('prev loss:{}'.format(prev_loss.item()))
        
        H += torch.eye(H.shape[0]).to(device)*alpha

        delta_w = -1 * torch.matmul(torch.inverse(H), g).detach()
        offset = 0
        for p in self._params:
            numel = p.numel()
            with torch.no_grad():
                p.add_(delta_w[offset:offset + numel].view_as(p),alpha=lr)
            offset += numel

        outputs, loss = closure(sample=False)
        print ('loss:{}'.format(loss.item()))

        if loss < prev_loss:
            print ('successful iteration')
            if alpha > 1e-5:
                group['alpha'] /= 10
        else:
            print ('failed iteration')
            if alpha < 1e5:
                group['alpha'] *= 10
            # undo the step
            offset = 0
            for p in self._params:    
                numel = p.numel()
                with torch.no_grad():
                    p.sub_(delta_w[offset:offset + numel].view_as(p),alpha=lr)
                offset += numel

        return outputs, record_loss
