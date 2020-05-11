import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import matplotlib.pyplot as plt

import numpy as np
import imageio

torch.manual_seed(1)    # reproducible

x = torch.unsqueeze(torch.linspace(-10, 10, 1000), dim=1)  # x data (tensor), shape=(100, 1)
y = torch.sin(x) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

# torch can only train on Variable, so convert them to Variable
x, y = Variable(x).to('cuda:0'), Variable(y).to('cuda:0')
# plt.figure(figsize=(10,4))
# plt.scatter(x.data.numpy(), y.data.numpy(), color = "blue")
# plt.title('Regression Analysis')
# plt.xlabel('Independent varible')
# plt.ylabel('Dependent varible')
# plt.savefig('curve_2.png')
# plt.show()

# 
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.elu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

# use the same net as before      
net = Net(n_feature=1, n_hidden=10, n_output=1)     # define the network
net.to('cuda:0')
print(net)  # net architecture
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

my_images = []
fig, ax = plt.subplots(figsize=(16,10))

# start training
for t in range(20000):
  
    prediction = net(x)     # input x and predict based on x

    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients
    
    # if t % 10 == 0:
    #     # plot and show learning process
    #     plt.cla()
    #     ax.set_title('Regression Analysis - model 1', fontsize=35)
    #     ax.set_xlabel('Independent variable', fontsize=24)
    #     ax.set_ylabel('Dependent variable', fontsize=24)
    #     ax.set_xlim(-11.0, 13.0)
    #     ax.set_ylim(-1.1, 1.2)
    #     ax.scatter(x.data.numpy(), y.data.numpy(), color = "blue")
    #     ax.plot(x.data.numpy(), prediction.data.numpy(), 'g-', lw=3)
    #     ax.text(8.8, -0.8, 'Step = %d' % t, fontdict={'size': 24, 'color':  'red'})
    #     ax.text(8.8, -0.95, 'Loss = %.4f' % loss.data.numpy(),
    #             fontdict={'size': 24, 'color':  'red'})

    #     # Used to return the plot as an image array 
    #     # (https://ndres.me/post/matplotlib-animated-gifs-easily/)
    #     fig.canvas.draw()       # draw the canvas, cache the renderer
    #     image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    #     image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    #     my_images.append(image)
    if t %1000 == 0:
        print(loss)
    
    


# save images as a gif    
# imageio.mimsave('./curve_2_model_1.gif', my_images, fps=10)

fig, ax = plt.subplots(figsize=(16,10))
plt.cla()
ax.set_title('Regression Analysis - model 3, Batches', fontsize=35)
ax.set_xlabel('Independent variable', fontsize=24)
ax.set_ylabel('Dependent variable', fontsize=24)
ax.set_xlim(-11.0, 13.0)
ax.set_ylim(-1.1, 1.2)
ax.scatter(x.data.cpu().numpy(), y.data.cpu().numpy(), color = "blue", alpha=0.2)
prediction = net(x.cuda('cuda:0'))     # input x and predict based on x
ax.scatter(x.data.cpu().numpy(), prediction.data.cpu().numpy(), color='green', alpha=0.5)
plt.savefig('curve_2_model_3_batches_adam.png')