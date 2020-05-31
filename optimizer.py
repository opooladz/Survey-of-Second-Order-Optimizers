import torch
from torch.optim.optimizer import Optimizer
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from functools import reduce
import torch.nn.functional as F

device = torch.device('cuda:0')

class LM(Optimizer):
    '''
    Arguments:
        lr: learning rate (step size)
        alpha: the hyperparameter in the regularization
    '''
    def __init__(self, params, lr=1, alpha=1):
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

        loss = closure(sample=False)
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

        return record_loss


class HessianFree(torch.optim.Optimizer):
    """
    Implements the Hessian-free algorithm presented in `Training Deep and
    Recurrent Networks with Hessian-Free Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1)
        delta_decay (float, optional): Decay of the previous result of
            computing delta with conjugate gradient method for the
            initialization of the next conjugate gradient iteration
        damping (float, optional): Initial value of the Tikhonov damping
            coefficient. (default: 0.5)
        max_iter (int, optional): Maximum number of Conjugate-Gradient
            iterations (default: 50)
        use_gnm (bool, optional): Use the generalized Gauss-Newton matrix:
            probably solves the indefiniteness of the Hessian (Section 20.6)
        verbose (bool, optional): Print statements (debugging)
    .. _Training Deep and Recurrent Networks with Hessian-Free Optimization:
        https://doi.org/10.1007/978-3-642-35289-8_27
    """

    def __init__(self, params,
                 lr=1,
                 damping=0.5,
                 delta_decay=0.95,
                 cg_max_iter=100,
                 use_gnm=True,
                 verbose=False):

        if not (0.0 < lr <= 1):
            raise ValueError("Invalid lr: {}".format(lr))

        if not (0.0 < damping <= 1):
            raise ValueError("Invalid damping: {}".format(damping))

        if not cg_max_iter > 0:
            raise ValueError("Invalid cg_max_iter: {}".format(cg_max_iter))

        defaults = dict(alpha=lr,
                        damping=damping,
                        delta_decay=delta_decay,
                        cg_max_iter=cg_max_iter,
                        use_gnm=use_gnm,
                        verbose=verbose)
        super(HessianFree, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "HessianFree doesn't support per-parameter options (parameter groups)")

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

    def step(self, closure, b=None, M_inv=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable): A closure that re-evaluates the model
                and returns a tuple of the loss and the output.
            b (callable, optional): A closure that calculates the vector b in
                the minimization problem x^T . A . x + x^T b.
            M (callable, optional): The INVERSE preconditioner of A
        """
        assert len(self.param_groups) == 1

        group = self.param_groups[0]
        alpha = group['alpha']
        delta_decay = group['delta_decay']
        cg_max_iter = group['cg_max_iter']
        damping = group['damping']
        use_gnm = group['use_gnm']
        verbose = group['verbose']

        state = self.state[self._params[0]]
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)

        loss_before, output = closure()
        current_evals = 1
        state['func_evals'] += 1

        # Gather current parameters and respective gradients
        flat_params = parameters_to_vector(self._params)
        flat_grad = self._gather_flat_grad()

        # Define linear operator
        if use_gnm:
            # Generalized Gauss-Newton vector product
            def A(x):
                return self._Gv(loss_before, output, x, damping)
        else:
            # Hessian-vector product
            def A(x):
                return self._Hv(flat_grad, x, damping)

        if M_inv is not None:
            m_inv = M_inv()

            # Preconditioner recipe (Section 20.13)
            if m_inv.dim() == 1:
                m = (m_inv + damping) ** (-0.85)

                def M(x):
                    return m * x
            else:
                m = torch.inverse(m_inv + damping * torch.eye(*m_inv.shape,device='cuda:0',dtype=torch.float64))

                def M(x):
                    return m @ x
        else:
            M = None

        b = flat_grad.detach() if b is None else b().detach().flatten()

        # Initializing Conjugate-Gradient (Section 20.10)
        if state.get('init_delta') is not None:
            init_delta = delta_decay * state.get('init_delta')
        else:
            init_delta = torch.zeros_like(flat_params)

        eps = torch.finfo(b.dtype).eps

        # Conjugate-Gradient
        deltas, Ms = self._CG(A=A, b=b.neg(), x0=init_delta,
                              M=M, max_iter=cg_max_iter,
                              tol=1e1 * eps, eps=eps, martens=True)

        # Update parameters
        delta = state['init_delta'] = deltas[-1]
        M = Ms[-1]

        vector_to_parameters(flat_params + delta, self._params)
        loss_now = closure()[0]
        current_evals += 1
        state['func_evals'] += 1

        # Conjugate-Gradient backtracking (Section 20.8.7)
        if verbose:
            print("Loss before CG: {}".format(float(loss_before)))
            print("Loss before BT: {}".format(float(loss_now)))

        for (d, m) in zip(reversed(deltas[:-1][::2]), reversed(Ms[:-1][::2])):
            vector_to_parameters(flat_params + d, self._params)
            loss_prev = closure()[0]
            if float(loss_prev) > float(loss_now):
                break
            delta = d
            M = m
            loss_now = loss_prev

        if verbose:
            print("Loss after BT:  {}".format(float(loss_now)))

        # The Levenberg-Marquardt Heuristic (Section 20.8.5)
        reduction_ratio = (float(loss_now) -
                           float(loss_before)) / M if M != 0 else 1

        if reduction_ratio < 0.25:
            group['damping'] *= 3 / 2
        elif reduction_ratio > 0.75:
            group['damping'] *= 2 / 3
        if reduction_ratio < 0:
            group['init_delta'] = 0

        # Line Searching (Section 20.8.8)
        beta = 0.8
        c = 1e-2
        min_improv = min(c * torch.dot(b, delta), 0)

        for _ in range(60):
            if float(loss_now) <= float(loss_before) + alpha * min_improv:
                break

            alpha *= beta
            vector_to_parameters(flat_params + alpha * delta, self._params)
            loss_now = closure()[0]
        else:  # No good update found
            alpha = 0.0
            loss_now = loss_before

        # Update the parameters (this time fo real)
        vector_to_parameters(flat_params + alpha * delta, self._params)

        if verbose:
            print("Loss after LS:  {0} (lr: {1:.3f})".format(
                float(loss_now), alpha))
            print("Tikhonov damping: {0:.3f} (reduction ratio: {1:.3f})".format(
                group['damping'], reduction_ratio), end='\n\n')

        return loss_now

    def _CG(self, A, b, x0, M=None, max_iter=50, tol=1.2e-6, eps=1.2e-7,
            martens=False):
        """
        Minimizes the linear system x^T.A.x - x^T b using the conjugate
            gradient method
        Arguments:
            A (callable): An abstract linear operator implementing the
                product A.x. A must represent a hermitian, positive definite
                matrix.
            b (torch.Tensor): The vector b.
            x0 (torch.Tensor): An initial guess for x.
            M (callable, optional): An abstract linear operator implementing
            the product of the preconditioner (for A) matrix with a vector.
            tol (float, optional): Tolerance for convergence.
            martens (bool, optional): Flag for Martens' convergence criterion.
        """

        x = [x0]
        r = A(x[0]) - b

        if M is not None:
            y = M(r)
            p = -y
        else:
            p = -r

        res_i_norm = r @ r

        if martens:
            m = [0.5 * (r - b) @ x0]

        for i in range(max_iter):
            Ap = A(p)

            alpha = res_i_norm / ((p @ Ap) + eps)

            x.append(x[i] + alpha * p)
            r = r + alpha * Ap

            if M is not None:
                y = M(r)
                res_ip1_norm = y @ r
            else:
                res_ip1_norm = r @ r

            beta = res_ip1_norm / (res_i_norm + eps)
            res_i_norm = res_ip1_norm

            # Martens' Relative Progress stopping condition (Section 20.4)
            if martens:
                m.append(0.5 * A(x[i + 1]) @ x[i + 1] - b @ x[i + 1])

                k = max(10, int(i / 10))
                if i > k:
                    stop = (m[i] - m[i - k]) / (m[i] + eps)
                    if stop < 1e-4:
                        break

            if res_i_norm < tol or torch.isnan(res_i_norm):
                break

            if M is not None:
                p = - y + beta * p
            else:
                p = - r + beta * p

        return (x, m) if martens else (x, None)

    def _Hv(self, gradient, vec, damping):
        """
        Computes the Hessian vector product.
        """
        Hv = self._Rop(gradient, self._params, vec)

        # Tikhonov damping (Section 20.8.1)
        return Hv.detach() + damping * vec

    def _Gv(self, loss, output, vec, damping):
        """
        Computes the generalized Gauss-Newton vector product.
        """
        Jv = self._Rop(output, self._params, vec)

        gradient = torch.autograd.grad(loss, output, create_graph=True)
        HJv = self._Rop(gradient, output, Jv)

        JHJv = torch.autograd.grad(
            output, self._params, grad_outputs=HJv.reshape_as(output), retain_graph=True)

        # Tikhonov damping (Section 20.8.1)
        return parameters_to_vector(JHJv).detach() + damping * vec

    @staticmethod
    def _Rop(y, x, v, create_graph=False):
        """
        Computes the product (dy_i/dx_j) v_j: R-operator
        """
        if isinstance(y, tuple):
            ws = [torch.zeros_like(y_i, requires_grad=True) for y_i in y]
        else:
            ws = torch.zeros_like(y, requires_grad=True)

        jacobian = torch.autograd.grad(
            y, x, grad_outputs=ws, create_graph=True)

        Jv = torch.autograd.grad(parameters_to_vector(
            jacobian), ws, grad_outputs=v, create_graph=create_graph)

        return parameters_to_vector(Jv)


# The empirical Fisher diagonal (Section 20.11.3)
def empirical_fisher_diagonal(net, xs, ys, criterion):
    grads = list()
    for (x, y) in zip(xs, ys):
        fi = criterion(net(x), y)
        grads.append(torch.autograd.grad(fi, net.parameters(),
                                         retain_graph=False))

    vec = torch.cat([(torch.stack(p) ** 2).mean(0).detach().flatten()
                     for p in zip(*grads)])
    return vec


# The empirical Fisher matrix (Section 20.11.3)
def empirical_fisher_matrix(net, xs, ys, criterion):
    grads = list()
    for (x, y) in zip(xs, ys):
        fi = criterion(net(x), y)
        grad = torch.autograd.grad(fi, net.parameters(),
                                   retain_graph=False)
        grads.append(torch.cat([g.detach().flatten() for g in grad]))

    grads = torch.stack(grads)
    n_batch = grads.shape[0]
    return torch.einsum('ij,ik->jk', grads, grads) / n_batch

class EKFAC(Optimizer):

    def __init__(self, net, eps, sua=False, ra=False, update_freq=1,
                 alpha=.75):
        """ EKFAC Preconditionner for Linear and Conv2d layers.
        Computes the EKFAC of the second moment of the gradients.
        It works for Linear and Conv2d layers and silently skip other layers.
        Args:
            net (torch.nn.Module): Network to precondition.
            eps (float): Tikhonov regularization parameter for the inverses.
            sua (bool): Applies SUA approximation.
            ra (bool): Computes stats using a running average of averaged gradients
                instead of using a intra minibatch estimate
            update_freq (int): Perform inverses every update_freq updates.
            alpha (float): Running average parameter
        """
        self.eps = eps
        self.sua = sua
        self.ra = ra
        self.update_freq = update_freq
        self.alpha = alpha
        self.params = []
        self._fwd_handles = []
        self._bwd_handles = []
        self._iteration_counter = 0
        if not self.ra and self.alpha != 1.:
            raise NotImplementedError
        for mod in net.modules():
            mod_class = mod.__class__.__name__
            if mod_class in ['Linear', 'Conv2d']:
                handle = mod.register_forward_pre_hook(self._save_input)
                self._fwd_handles.append(handle)
                handle = mod.register_backward_hook(self._save_grad_output)
                self._bwd_handles.append(handle)
                params = [mod.weight]
                if mod.bias is not None:
                    params.append(mod.bias)
                d = {'params': params, 'mod': mod, 'layer_type': mod_class}
                if mod_class == 'Conv2d':
                    if not self.sua:
                        # Adding gathering filter for convolution
                        d['gathering_filter'] = self._get_gathering_filter(mod)
                self.params.append(d)
        super(EKFAC, self).__init__(self.params, {})

    def step(self, update_stats=True, update_params=True):
        """Performs one step of preconditioning."""
        for group in self.param_groups:
            # Getting parameters
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None
            state = self.state[weight]
            # Update convariances and inverses
            if self._iteration_counter % self.update_freq == 0:
                self._compute_kfe(group, state)
            # Preconditionning
            if group['layer_type'] == 'Conv2d' and self.sua:
                if self.ra:
                    self._precond_sua_ra(weight, bias, group, state)
                else:
                    self._precond_intra_sua(weight, bias, group, state)
            else:
                if self.ra:
                    self._precond_ra(weight, bias, group, state)
                else:
                    self._precond_intra(weight, bias, group, state)
        self._iteration_counter += 1

    def _save_input(self, mod, i):
        """Saves input of layer to compute covariance."""
        self.state[mod]['x'] = i[0]

    def _save_grad_output(self, mod, grad_input, grad_output):
        """Saves grad on output of layer to compute covariance."""
        self.state[mod]['gy'] = grad_output[0] * grad_output[0].size(0)

    def _precond_ra(self, weight, bias, group, state):
        """Applies preconditioning."""
        kfe_x = state['kfe_x']
        kfe_gy = state['kfe_gy']
        m2 = state['m2']
        g = weight.grad.data
        s = g.shape
        bs = self.state[group['mod']]['x'].size(0)
        if group['layer_type'] == 'Conv2d':
            g = g.contiguous().view(s[0], s[1]*s[2]*s[3])
        if bias is not None:
            gb = bias.grad.data
            g = torch.cat([g, gb.view(gb.shape[0], 1)], dim=1)
        g_kfe = torch.mm(torch.mm(kfe_gy.t(), g), kfe_x)
        m2.mul_(self.alpha).add_((1. - self.alpha) * bs, g_kfe**2)
        g_nat_kfe = g_kfe / (m2 + self.eps)
        g_nat = torch.mm(torch.mm(kfe_gy, g_nat_kfe), kfe_x.t())
        if bias is not None:
            gb = g_nat[:, -1].contiguous().view(*bias.shape)
            bias.grad.data = gb
            g_nat = g_nat[:, :-1]
        g_nat = g_nat.contiguous().view(*s)
        weight.grad.data = g_nat

    def _precond_intra(self, weight, bias, group, state):
        """Applies preconditioning."""
        kfe_x = state['kfe_x']
        kfe_gy = state['kfe_gy']
        mod = group['mod']
        x = self.state[mod]['x']
        gy = self.state[mod]['gy']
        g = weight.grad.data
        s = g.shape
        s_x = x.size()
        s_cin = 0
        s_gy = gy.size()
        bs = x.size(0)
        if group['layer_type'] == 'Conv2d':
            x = F.conv2d(x, group['gathering_filter'],
                         stride=mod.stride, padding=mod.padding,
                         groups=mod.in_channels)
            s_x = x.size()
            x = x.data.permute(1, 0, 2, 3).contiguous().view(x.shape[1], -1)
            if mod.bias is not None:
                ones = torch.ones_like(x[:1])
                x = torch.cat([x, ones], dim=0)
                s_cin = 1 # adding a channel in dim for the bias
            # intra minibatch m2
            x_kfe = torch.mm(kfe_x.t(), x).view(s_x[1]+s_cin, -1, s_x[2], s_x[3]).permute(1, 0, 2, 3)
            gy = gy.permute(1, 0, 2, 3).contiguous().view(s_gy[1], -1)
            gy_kfe = torch.mm(kfe_gy.t(), gy).view(s_gy[1], -1, s_gy[2], s_gy[3]).permute(1, 0, 2, 3)
            m2 = torch.zeros((s[0], s[1]*s[2]*s[3]+s_cin), device=g.device)
            g_kfe = torch.zeros((s[0], s[1]*s[2]*s[3]+s_cin), device=g.device)
            for i in range(x_kfe.size(0)):
                g_this = torch.mm(gy_kfe[i].view(s_gy[1], -1),
                                  x_kfe[i].permute(1, 2, 0).view(-1, s_x[1]+s_cin))
                m2 += g_this**2
            m2 /= bs
            g_kfe = torch.mm(gy_kfe.permute(1, 0, 2, 3).view(s_gy[1], -1),
                             x_kfe.permute(0, 2, 3, 1).contiguous().view(-1, s_x[1]+s_cin)) / bs
            ## sanity check did we obtain the same grad ?
            # g = torch.mm(torch.mm(kfe_gy, g_kfe), kfe_x.t())
            # gb = g[:,-1]
            # gw = g[:,:-1].view(*s)
            # print('bias', torch.dist(gb, bias.grad.data))
            # print('weight', torch.dist(gw, weight.grad.data))
            ## end sanity check
            g_nat_kfe = g_kfe / (m2 + self.eps)
            g_nat = torch.mm(torch.mm(kfe_gy, g_nat_kfe), kfe_x.t())
            if bias is not None:
                gb = g_nat[:, -1].contiguous().view(*bias.shape)
                bias.grad.data = gb
                g_nat = g_nat[:, :-1]
            g_nat = g_nat.contiguous().view(*s)
            weight.grad.data = g_nat
        else:
            if bias is not None:
                ones = torch.ones_like(x[:, :1])
                x = torch.cat([x, ones], dim=1)
            x_kfe = torch.mm(x, kfe_x)
            gy_kfe = torch.mm(gy, kfe_gy)
            m2 = torch.mm(gy_kfe.t()**2, x_kfe**2) / bs
            g_kfe = torch.mm(gy_kfe.t(), x_kfe) / bs
            g_nat_kfe = g_kfe / (m2 + self.eps)
            g_nat = torch.mm(torch.mm(kfe_gy, g_nat_kfe), kfe_x.t())
            if bias is not None:
                gb = g_nat[:, -1].contiguous().view(*bias.shape)
                bias.grad.data = gb
                g_nat = g_nat[:, :-1]
            g_nat = g_nat.contiguous().view(*s)
            weight.grad.data = g_nat

    def _precond_sua_ra(self, weight, bias, group, state):
        """Preconditioning for KFAC SUA."""
        kfe_x = state['kfe_x']
        kfe_gy = state['kfe_gy']
        m2 = state['m2']
        g = weight.grad.data
        s = g.shape
        bs = self.state[group['mod']]['x'].size(0)
        mod = group['mod']
        if bias is not None:
            gb = bias.grad.view(-1, 1, 1, 1).expand(-1, -1, s[2], s[3])
            g = torch.cat([g, gb], dim=1)
        g_kfe = self._to_kfe_sua(g, kfe_x, kfe_gy)
        m2.mul_(self.alpha).add_((1. - self.alpha) * bs, g_kfe**2)
        g_nat_kfe = g_kfe / (m2 + self.eps)
        g_nat = self._to_kfe_sua(g_nat_kfe, kfe_x.t(), kfe_gy.t())
        if bias is not None:
            gb = g_nat[:, -1, s[2]//2, s[3]//2]
            bias.grad.data = gb
            g_nat = g_nat[:, :-1]
        weight.grad.data = g_nat

    def _precond_intra_sua(self, weight, bias, group, state):
        """Preconditioning for KFAC SUA."""
        kfe_x = state['kfe_x']
        kfe_gy = state['kfe_gy']
        mod = group['mod']
        x = self.state[mod]['x']
        gy = self.state[mod]['gy']
        g = weight.grad.data
        s = g.shape
        s_x = x.size()
        s_gy = gy.size()
        s_cin = 0
        bs = x.size(0)
        if bias is not None:
            ones = torch.ones_like(x[:,:1])
            x = torch.cat([x, ones], dim=1)
            s_cin += 1
        # intra minibatch m2
        x = x.permute(1, 0, 2, 3).contiguous().view(s_x[1]+s_cin, -1)
        x_kfe = torch.mm(kfe_x.t(), x).view(s_x[1]+s_cin, -1, s_x[2], s_x[3]).permute(1, 0, 2, 3)
        gy = gy.permute(1, 0, 2, 3).contiguous().view(s_gy[1], -1)
        gy_kfe = torch.mm(kfe_gy.t(), gy).view(s_gy[1], -1, s_gy[2], s_gy[3]).permute(1, 0, 2, 3)
        m2 = torch.zeros((s[0], s[1]+s_cin, s[2], s[3]), device=g.device)
        g_kfe = torch.zeros((s[0], s[1]+s_cin, s[2], s[3]), device=g.device)
        for i in range(x_kfe.size(0)):
            g_this = grad_wrt_kernel(x_kfe[i:i+1], gy_kfe[i:i+1], mod.padding, mod.stride)
            m2 += g_this**2
        m2 /= bs
        g_kfe = grad_wrt_kernel(x_kfe, gy_kfe, mod.padding, mod.stride) / bs
        ## sanity check did we obtain the same grad ?
        # g = self._to_kfe_sua(g_kfe, kfe_x.t(), kfe_gy.t())
        # gb = g[:, -1, s[2]//2, s[3]//2]
        # gw = g[:,:-1].view(*s)
        # print('bias', torch.dist(gb, bias.grad.data))
        # print('weight', torch.dist(gw, weight.grad.data))
        ## end sanity check
        g_nat_kfe = g_kfe / (m2 + self.eps)
        g_nat = self._to_kfe_sua(g_nat_kfe, kfe_x.t(), kfe_gy.t())
        if bias is not None:
            gb = g_nat[:, -1, s[2]//2, s[3]//2]
            bias.grad.data = gb
            g_nat = g_nat[:, :-1]
        weight.grad.data = g_nat

    def _compute_kfe(self, group, state):
        """Computes the covariances."""
        mod = group['mod']
        x = self.state[group['mod']]['x']
        gy = self.state[group['mod']]['gy']
        # Computation of xxt
        if group['layer_type'] == 'Conv2d':
            if not self.sua:
                x = F.conv2d(x, group['gathering_filter'],
                             stride=mod.stride, padding=mod.padding,
                             groups=mod.in_channels)
            x = x.data.permute(1, 0, 2, 3).contiguous().view(x.shape[1], -1)
        else:
            x = x.data.t()
        if mod.bias is not None:
            ones = torch.ones_like(x[:1])
            x = torch.cat([x, ones], dim=0)
        xxt = torch.mm(x, x.t()) / float(x.shape[1])
        Ex, state['kfe_x'] = torch.symeig(xxt, eigenvectors=True)
        # Computation of ggt
        if group['layer_type'] == 'Conv2d':
            gy = gy.data.permute(1, 0, 2, 3)
            state['num_locations'] = gy.shape[2] * gy.shape[3]
            gy = gy.contiguous().view(gy.shape[0], -1)
        else:
            gy = gy.data.t()
            state['num_locations'] = 1
        ggt = torch.mm(gy, gy.t()) / float(gy.shape[1])
        Eg, state['kfe_gy'] = torch.symeig(ggt, eigenvectors=True)
        state['m2'] = Eg.unsqueeze(1) * Ex.unsqueeze(0) * state['num_locations']
        if group['layer_type'] == 'Conv2d' and self.sua:
            ws = group['params'][0].grad.data.size()
            state['m2'] = state['m2'].view(Eg.size(0), Ex.size(0), 1, 1).expand(-1, -1, ws[2], ws[3])

    def _get_gathering_filter(self, mod):
        """Convolution filter that extracts input patches."""
        kw, kh = mod.kernel_size
        g_filter = mod.weight.data.new(kw * kh * mod.in_channels, 1, kw, kh)
        g_filter.fill_(0)
        for i in range(mod.in_channels):
            for j in range(kw):
                for k in range(kh):
                    g_filter[k + kh*j + kw*kh*i, 0, j, k] = 1
        return g_filter

    def _to_kfe_sua(self, g, vx, vg):
        """Project g to the kfe"""
        sg = g.size()
        g = torch.mm(vg.t(), g.view(sg[0], -1)).view(vg.size(1), sg[1], sg[2], sg[3])
        g = torch.mm(g.permute(0, 2, 3, 1).contiguous().view(-1, sg[1]), vx)
        g = g.view(vg.size(1), sg[2], sg[3], vx.size(1)).permute(0, 3, 1, 2)
        return g

    def __del__(self):
        for handle in self._fwd_handles + self._bwd_handles:
            handle.remove()


def grad_wrt_kernel(a, g, padding, stride, target_size=None):
    gk = F.conv2d(a.transpose(0, 1), g.transpose(0, 1).contiguous(),
                  padding=padding, dilation=stride).transpose(0, 1)
    if target_size is not None and target_size != gk.size():
        return gk[:, :, :target_size[2], :target_size[3]].contiguous()
    return gk

class KFAC(Optimizer):

    def __init__(self, net, eps, sua=False, pi=False, update_freq=1,
                 alpha=1.0, constraint_norm=False):
        """ K-FAC Preconditionner for Linear and Conv2d layers.
        Computes the K-FAC of the second moment of the gradients.
        It works for Linear and Conv2d layers and silently skip other layers.
        Args:
            net (torch.nn.Module): Network to precondition.
            eps (float): Tikhonov regularization parameter for the inverses.
            sua (bool): Applies SUA approximation.
            pi (bool): Computes pi correction for Tikhonov regularization.
            update_freq (int): Perform inverses every update_freq updates.
            alpha (float): Running average parameter (if == 1, no r. ave.).
            constraint_norm (bool): Scale the gradients by the squared
                fisher norm.
        """
        self.eps = eps
        self.sua = sua
        self.pi = pi
        self.update_freq = update_freq
        self.alpha = alpha
        self.constraint_norm = constraint_norm
        self.params = []
        self._fwd_handles = []
        self._bwd_handles = []
        self._iteration_counter = 0
        for mod in net.modules():
            mod_class = mod.__class__.__name__
            if mod_class in ['Linear', 'Conv2d']:
                handle = mod.register_forward_pre_hook(self._save_input)
                self._fwd_handles.append(handle)
                handle = mod.register_backward_hook(self._save_grad_output)
                self._bwd_handles.append(handle)
                params = [mod.weight]
                if mod.bias is not None:
                    params.append(mod.bias)
                d = {'params': params, 'mod': mod, 'layer_type': mod_class}
                self.params.append(d)
        super(KFAC, self).__init__(self.params, {})

    def step(self, update_stats=True, update_params=True):
        """Performs one step of preconditioning."""
        fisher_norm = 0.
        for group in self.param_groups:
            # Getting parameters
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None
            state = self.state[weight]
            # Update convariances and inverses
            if update_stats:
                if self._iteration_counter % self.update_freq == 0:
                    self._compute_covs(group, state)
                    ixxt, iggt = self._inv_covs(state['xxt'], state['ggt'],
                                                state['num_locations'])
                    state['ixxt'] = ixxt
                    state['iggt'] = iggt
                else:
                    if self.alpha != 1:
                        self._compute_covs(group, state)
            if update_params:
                # Preconditionning
                gw, gb = self._precond(weight, bias, group, state)
                # Updating gradients
                if self.constraint_norm:
                    fisher_norm += (weight.grad * gw).sum()
                weight.grad.data = gw
                if bias is not None:
                    if self.constraint_norm:
                        fisher_norm += (bias.grad * gb).sum()
                    bias.grad.data = gb
            # Cleaning
            if 'x' in self.state[group['mod']]:
                del self.state[group['mod']]['x']
            if 'gy' in self.state[group['mod']]:
                del self.state[group['mod']]['gy']
        # Eventually scale the norm of the gradients
        if update_params and self.constraint_norm:
            scale = (1. / fisher_norm) ** 0.5
            for group in self.param_groups:
                for param in group['params']:
                    param.grad.data *= scale
        if update_stats:
            self._iteration_counter += 1

    def _save_input(self, mod, i):
        """Saves input of layer to compute covariance."""
        if mod.training:
            self.state[mod]['x'] = i[0]

    def _save_grad_output(self, mod, grad_input, grad_output):
        """Saves grad on output of layer to compute covariance."""
        if mod.training:
            self.state[mod]['gy'] = grad_output[0] * grad_output[0].size(0)

    def _precond(self, weight, bias, group, state):
        """Applies preconditioning."""
        if group['layer_type'] == 'Conv2d' and self.sua:
            return self._precond_sua(weight, bias, group, state)
        ixxt = state['ixxt']
        iggt = state['iggt']
        g = weight.grad.data
        s = g.shape
        if group['layer_type'] == 'Conv2d':
            g = g.contiguous().view(s[0], s[1]*s[2]*s[3])
        if bias is not None:
            gb = bias.grad.data
            g = torch.cat([g, gb.view(gb.shape[0], 1)], dim=1)
        g = torch.mm(torch.mm(iggt, g), ixxt)
        if group['layer_type'] == 'Conv2d':
            g /= state['num_locations']
        if bias is not None:
            gb = g[:, -1].contiguous().view(*bias.shape)
            g = g[:, :-1]
        else:
            gb = None
        g = g.contiguous().view(*s)
        return g, gb

    def _precond_sua(self, weight, bias, group, state):
        """Preconditioning for KFAC SUA."""
        ixxt = state['ixxt']
        iggt = state['iggt']
        g = weight.grad.data
        s = g.shape
        mod = group['mod']
        g = g.permute(1, 0, 2, 3).contiguous()
        if bias is not None:
            gb = bias.grad.view(1, -1, 1, 1).expand(1, -1, s[2], s[3])
            g = torch.cat([g, gb], dim=0)
        g = torch.mm(ixxt, g.contiguous().view(-1, s[0]*s[2]*s[3]))
        g = g.view(-1, s[0], s[2], s[3]).permute(1, 0, 2, 3).contiguous()
        g = torch.mm(iggt, g.view(s[0], -1)).view(s[0], -1, s[2], s[3])
        g /= state['num_locations']
        if bias is not None:
            gb = g[:, -1, s[2]//2, s[3]//2]
            g = g[:, :-1]
        else:
            gb = None
        return g, gb

    def _compute_covs(self, group, state):
        """Computes the covariances."""
        mod = group['mod']
        x = self.state[group['mod']]['x']
        gy = self.state[group['mod']]['gy']
        # Computation of xxt
        if group['layer_type'] == 'Conv2d':
            if not self.sua:
                x = F.unfold(x, mod.kernel_size, padding=mod.padding,
                             stride=mod.stride)
            else:
                x = x.view(x.shape[0], x.shape[1], -1)
            x = x.data.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
        else:
            x = x.data.t()
        if mod.bias is not None:
            ones = torch.ones_like(x[:1])
            x = torch.cat([x, ones], dim=0)
        if self._iteration_counter == 0:
            state['xxt'] = torch.mm(x, x.t()) / float(x.shape[1])
        else:
            state['xxt'].addmm_(mat1=x, mat2=x.t(),
                                beta=(1. - self.alpha),
                                alpha=self.alpha / float(x.shape[1]))
        # Computation of ggt
        if group['layer_type'] == 'Conv2d':
            gy = gy.data.permute(1, 0, 2, 3)
            state['num_locations'] = gy.shape[2] * gy.shape[3]
            gy = gy.contiguous().view(gy.shape[0], -1)
        else:
            gy = gy.data.t()
            state['num_locations'] = 1
        if self._iteration_counter == 0:
            state['ggt'] = torch.mm(gy, gy.t()) / float(gy.shape[1])
        else:
            state['ggt'].addmm_(mat1=gy, mat2=gy.t(),
                                beta=(1. - self.alpha),
                                alpha=self.alpha / float(gy.shape[1]))

    def _inv_covs(self, xxt, ggt, num_locations):
        """Inverses the covariances."""
        # Computes pi
        pi = 1.0
        if self.pi:
            tx = torch.trace(xxt) * ggt.shape[0]
            tg = torch.trace(ggt) * xxt.shape[0]
            pi = (tx / tg)
        # Regularizes and inverse
        eps = self.eps / num_locations
        diag_xxt = xxt.new(xxt.shape[0]).fill_((eps * pi) ** 0.5)
        diag_ggt = ggt.new(ggt.shape[0]).fill_((eps / pi) ** 0.5)
        ixxt = (xxt + torch.diag(diag_xxt)).inverse()
        iggt = (ggt + torch.diag(diag_ggt)).inverse()
        return ixxt, iggt
    
    def __del__(self):
        for handle in self._fwd_handles + self._bwd_handles:
            handle.remove()    