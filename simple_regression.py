import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import matplotlib.pyplot as plt

import numpy as np
import imageio
import numpy as np
import functools
import matplotlib.pyplot as plt
# from .optimizer import Optimizer, required

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms
import sys
# sys.path.append('/home/npcusero/Documents/notclassified/PyTorch-LBFGS/functions')
sys.path.append('/home/npcusero/Documents/notclassified/optimizer/hessian_jacobian')

# import hessian
import gradient 
import torch.optim as optim





torch.manual_seed(1)    # reproducible

x = torch.unsqueeze(torch.linspace(-10, 10, 1000), dim=1)  # x data (tensor), shape=(100, 1)
y = torch.sin(x) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

# torch can only train on Variable, so convert them to Variable
x, y = Variable(x), Variable(y)
plt.figure(figsize=(10,4))
plt.scatter(x.data.numpy(), y.data.numpy(), color = "blue")
plt.title('Regression Analysis')
plt.xlabel('Independent varible')
plt.ylabel('Dependent varible')
plt.savefig('curve_2.png')
plt.show()


def LM(model,loss,n_iter=10):

    alpha=1e-3
    loss_hist=[]
    running_loss = 0.0
    for epoch in range(n_iter):
        flag = True
        for i, (data, target) in enumerate(torch_dataset):
            # put network in training mode
            model.train()
            # move data to GPU
            data, target = data.cuda('cuda:0'), target.cuda('cuda:0')
            #Variables in Pytorch are differenciable. 
            data, target = Variable(data), Variable(target)
            # target = target.type(torch.float32)
            # run data through network
            out=model(data)
            # loss_out=loss(out,target)
            # calculate loss 
            loss_tmp=loss(out,target)
            loss_out = loss_tmp.mean()
            prev_loss=loss_out.item()

            # gradients=torch.autograd.grad(loss_out, model.parameters(), create_graph=True)
            # Caluclate Jacobian 
            J = gradient.jacobian(loss_tmp,model.parameters(), create_graph=True, retain_graph=True)


            model.eval()

            # Hessian, g_vector=eval_hessian(gradients, model)
            # g_vector = eval_gvec(gradients,model)

            # Hessian, g_vector =  get_hessian(gradients, model)
            # J = gradient.jacobian(loss_tmp,model.parameters(), create_graph=True)

            # Calculate GN matrix 
            Hessian = 2*torch.matmul(J.T,J)
            g_vector = torch.matmul(J.T,loss_tmp).unsqueeze(1)
            dx=-1*torch.matmul(torch.inverse((alpha*torch.eye(Hessian.shape[-1]).cuda('cuda:0')+Hessian)),g_vector).detach()
            # dont use this version since i always get a singular matrix error 
            # dx=-1*(alpha*torch.diag(Hessian).cuda('cuda:0')+Hessian).inverse().mm(g_vector).detach()
            
            # cnt logic has to do with LM method
            # not 100% sure about it to be honest 
            cnt=0
            # set gradients to zero before doing back prop because PyTprch 
            # accumulates the grads on subsequent back passes 
            model.zero_grad()

            #  loop thorugh the model params
            for p in model.parameters():
                mm=torch.Tensor([p.shape]).tolist()[0]
                num=int(functools.reduce(lambda x,y:x*y,mm,1))
                # make pytorch stop calculating the computational graph 
                # becuase about to do update step
                p.requires_grad=False
                # not 100% about the cnt logic
                p+=dx[cnt:cnt+num,:].reshape(p.shape)
                cnt+=num
                p.requires_grad=True


            out=model(data)#.unsqueeze(1)
            loss_out=loss(out,target)
            # these are the /10 and *10 rules you had requested
            if loss_out<prev_loss:
                #print("Successful iteration")
                # print(loss_out.data)
                #loss_hist.append(loss_out)
                if flag:
                    # plot and show learning process
                    plt.cla()
                    ax.set_title('Regression Analysis - model 3 Batches', fontsize=35)
                    ax.set_xlabel('Independent variable', fontsize=24)
                    ax.set_ylabel('Dependent variable', fontsize=24)
                    ax.set_xlim(-11.0, 13.0)
                    ax.set_ylim(-1.1, 1.2)
                    ax.scatter(data.data.cpu().numpy(), target.data.cpu().numpy(), color = "blue", alpha=0.2)
                    ax.scatter(data.data.cpu().numpy(), out.data.cpu().numpy(), color='green', alpha=0.5)
                    ax.text(8.8, -0.8, 'Epoch = %d' % epoch,
                            fontdict={'size': 24, 'color':  'red'})
                    ax.text(8.8, -0.95, 'Loss = %.4f' % loss_out.data.cpu().numpy(),
                            fontdict={'size': 24, 'color':  'red'})

                    # Used to return the plot as an image array 
                    # (https://ndres.me/post/matplotlib-animated-gifs-easily/)
                    fig.canvas.draw()       # draw the canvas, cache the renderer
                    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                    my_images.append(image)    
                    flag = False
                    
                if alpha > 1e-5:
                    alpha/=10
                # print('alpha: {}' .format(alpha))
            else:
                #print("Augmenting step size")
                if alpha < 1e5:
                    alpha*=10
                #print('alpha: {}' .format(alpha))
                cnt=0
                for p in model.parameters():
                    mm=torch.Tensor([p.shape]).tolist()[0]
                    num=int(functools.reduce(lambda x,y:x*y,mm,1))
                    p.requires_grad=False
                    p-=dx[cnt:cnt+num,:].reshape(p.shape)
                    cnt+=num
                    p.requires_grad=True

            # print('Loss {}' .format(loss_out))
            # print statistics
            running_loss += loss_out
            # print(running_loss)
            if i % 20 == 19:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
            # if i == 1:
                # # plot and show learning process
                # plt.cla()
                # ax.set_title('Regression Analysis - model 3 Batches', fontsize=35)
                # ax.set_xlabel('Independent variable', fontsize=24)
                # ax.set_ylabel('Dependent variable', fontsize=24)
                # ax.set_xlim(-11.0, 13.0)
                # ax.set_ylim(-1.1, 1.2)
                # ax.scatter(data.data.cpu().numpy(), target.data.cpu().numpy(), color = "blue", alpha=0.2)
                # ax.scatter(data.data.cpu().numpy(), out.data.cpu().numpy(), color='green', alpha=0.5)
                # ax.text(8.8, -0.8, 'Epoch = %d' % epoch,
                #         fontdict={'size': 24, 'color':  'red'})
                # ax.text(8.8, -0.95, 'Loss = %.4f' % loss_out.data.cpu().numpy(),
                #         fontdict={'size': 24, 'color':  'red'})

                # # Used to return the plot as an image array 
                # # (https://ndres.me/post/matplotlib-animated-gifs-easily/)
                # fig.canvas.draw()       # draw the canvas, cache the renderer
                # image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                # image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                # my_images.append(image)                     

    return loss_hist, my_images 

    
def get_hessian(loss_grad,model):
    cnt = 0
    for g in loss_grad:
        g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector,     g.contiguous().view(-1)])
        cnt = 1
    J = gradient.jacobian(g_vector[0],model.parameters(),create_graph=True)
    return 2*torch.matmul(J.T,J) , g_vector.unsqueeze(1)

# def jacobian_pytorch(model):
#     jacobian_mtx = []
#     for m in range(M):
#         # We iterate over the M elements of the output vector
#         grad_func = torch.autograd.grad(model.output[:, m], model.input)
#         gradients = sess.run(grad_func, feed_dict={model.input: x.reshape((1, x.size))})


def eval_gvec(loss_grad, model):
    cnt = 0
    for g in loss_grad:
        g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector,     g.contiguous().view(-1)])
        cnt = 1
    return g_vector.unsqueeze(1)

def eval_hessian(loss_grad, model):
    cnt = 0
    for g in loss_grad:
        g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector,     g.contiguous().view(-1)])
        cnt = 1
    l = g_vector.size(0)
    hessian = torch.zeros(l, l).cuda('cuda:0')
    for idx in range(l):
        grad2rd = torch.autograd.grad(g_vector[idx], model.parameters(), create_graph=True)
        cnt = 0
        for g in grad2rd:
            g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
            cnt = 1
        hessian[idx] = g2
    return hessian, g_vector.unsqueeze(1)

# another way to define a network
net = torch.nn.Sequential(
        torch.nn.Linear(1, 200),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(200, 100),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(100, 1),
    )
net.cuda('cuda:0')

# optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss(reduction='none')  # this is for regression mean squared loss

BATCH_SIZE = 64
EPOCH = 200

torch_dataset = Data.TensorDataset(x, y)

loader = Data.DataLoader(
    dataset=torch_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, num_workers=2,)

my_images = []
fig, ax = plt.subplots(figsize=(16,10))
loss_hist, my_images=LM(net,loss_func,n_iter=10)


# # save images as a gif    
imageio.mimsave('./curve_2_model_3_batch.gif', my_images, fps=12)


fig, ax = plt.subplots(figsize=(16,10))
plt.cla()
ax.set_title('Regression Analysis - model 3, Batches', fontsize=35)
ax.set_xlabel('Independent variable', fontsize=24)
ax.set_ylabel('Dependent variable', fontsize=24)
ax.set_xlim(-11.0, 13.0)
ax.set_ylim(-1.1, 1.2)
ax.scatter(x.data.numpy(), y.data.numpy(), color = "blue", alpha=0.2)
prediction = net(x.cuda('cuda:0'))     # input x and predict based on x
ax.scatter(x.data.cpu().numpy(), prediction.data.cpu().numpy(), color='green', alpha=0.5)
plt.savefig('curve_2_model_3_batches.png')
# plt.show()
