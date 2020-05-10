import torch
import numpy as np
import functools
import matplotlib.pyplot as plt

def LM(model,loss,n_iter=30):

    alpha=1e-3
    loss_hist=[]
    for i in range(n_iter):
        model.train()
        out=model(toy_input).unsqueeze(1)
        loss_out=loss(out)
        prev_loss=loss_out.item()
        gradients=torch.autograd.grad(loss_out, model.parameters(), create_graph=True)
        model.eval()
        Hessian, g_vector=eval_hessian(gradients, model)

        dx=-1(alpha*torch.eye(Hessian.shape[-1]).cuda()+Hessian).inverse().mm(g_vector).detach()

        cnt=0
        model.zero_grad()

        for p in model.parameters():

            mm=torch.Tensor([p.shape]).tolist()[0]
            num=int(functools.reduce(lambda x,y:x*y,mm,1))
            p.requires_grad=False
            p+=dx[cnt:cnt+num,:].reshape(p.shape)
            cnt+=num
            p.requires_grad=True


        out=model(toy_input).unsqueeze(1)
        loss_out=loss(out)

        if loss_out<prev_loss:
            print("Successful iteration")
            loss_hist.append(loss_out)
            alpha/=10
        else:
            print("Augmenting step size")
            alpha*=10
            cnt=0
            for p in model.parameters():

                mm=torch.Tensor([p.shape]).tolist()[0]
                num=int(functools.reduce(lambda x,y:x*y,mm,1))
                p.requires_grad=False
                p-=dx[cnt:cnt+num,:].reshape(p.shape)
                cnt+=num
                p.requires_grad=True

    return loss_hist 



def eval_hessian(loss_grad, model):
    cnt = 0
    for g in loss_grad:
        g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector,     g.contiguous().view(-1)])
        cnt = 1
    l = g_vector.size(0)
    hessian = torch.zeros(l, l).cuda()
    for idx in range(l):
        grad2rd = torch.autograd.grad(g_vector[idx], model.parameters(), create_graph=True)
        cnt = 0
        for g in grad2rd:
            g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
            cnt = 1
        hessian[idx] = g2
    return hessian, g_vector.unsqueeze(1)

def toy_loss(vec):
    return vec.transpose(0,1).mm(vec)

class toy_model(torch.nn.Module):

    def __init__(self,in_c,width,height):

        super().__init__()

        self.cnv=torch.nn.Conv2d(in_c,1,3,1,padding=1)
        self.lin=torch.nn.Linear(1*width*height,16)

    def forward(self,tns):

        out=self.cnv(tns)
        out=self.lin(out.view(-1))
        return out

if __name__=="__main__":

    H=20
    W=20
    toy_input=torch.rand(1,3,H,W).cuda()
    toy_mdl=toy_model(3,W,H)
    toy_mdl.cuda()

    loss_hist=LM(toy_mdl,lambda x:toy_loss(x))
