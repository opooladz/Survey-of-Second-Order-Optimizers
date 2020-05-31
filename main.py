from models import *
from gradient import *
from optimizer import *
from dataloading import *
import argparse
import torchvision

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist', type=str, choices=['cifar10', 'mnist', 'mnist_small', 'regression'])
parser.add_argument('--optimizer', default='LM', type=str, choices=['LM', 'SGD', 'Adam', 'HF', 'KFAC'])
parser.add_argument('--net_type', default='cnn', type=str, choices=['cnn', 'mlp'])
parser.add_argument('--epoch_num', default=1, type=int)
parser.add_argument('--device', default=0, type=int)
args = parser.parse_args()

device = torch.device('cuda:' + str(args.device))

print ('------------------loading data------------------')
if args.dataset == 'mnist':
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_MNIST)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_MNIST)
elif args.dataset == 'mnist_small':
    trainset = MNIST_small(train=True)
    testset = MNIST_small(train=False)
elif args.dataset == 'regression':
    pass

trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=0)
testloader = DataLoader(testset, batch_size=512, shuffle=False, num_workers=0)

print ('------------------initializating network----------------------')

if args.net_type == 'cnn':
    model = CNN().to(device)
elif args.net_type == 'mlp':
    model = MLP().to(device)

if args.optimizer == 'LM':
    optimizer = LM(model.parameters(), lr=0.5, alpha=10)
elif args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
elif args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

if args.dataset != 'regression':
    criterion= nn.CrossEntropyLoss()

id = args.dataset + '_' + args.net_type + '_' + args.optimizer 
print (id)

print ('-------------------training---------------------------')
train_loss = []
test_loss = []
train_acc = []
test_acc = []
for epoch in range(args.epoch_num):
    for idx, (inputs, targets) in enumerate (trainloader):
        print ('iter:{}'.format(idx + 1))
        inputs, targets = inputs.to(device), targets.to(device)
        N = inputs.shape[0]
        if args.optimizer=='LM':
            def closure(sample=True):
                N = inputs.shape[0]
                optimizer.zero_grad()
                if sample:
                    with save_sample_grads(model):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        g_all = gather_flat_grads(model.parameters())
                        g = g_all.sum(0)
                        H = N * torch.einsum('ijk, ikl -> ijl', [torch.unsqueeze(g_all, 2), torch.unsqueeze(g_all, 1)]).sum(0)
                        return loss, g, H 
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    return outputs, loss

            outputs, record_loss = optimizer.step(closure)
            correct = torch.sum(torch.argmax(outputs,1) == targets).item()
            
            train_acc.append(correct/N)
            train_loss.append(record_loss)
            if idx == 50:
                break        

        elif args.optimizer == 'Adam' or args.optimizer == 'SGD':
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,targets)
            loss.backward()
            optimizer.step()
            correct = torch.sum(torch.argmax(outputs,1) == targets).item()
            train_loss.append(loss.item())
            train_acc.append(correct/N)

            if idx == 50:
                break

        print ('testing...')
        total, correct = 0, 0
        running_loss = 0.0
        for idx, (inputs, targets) in enumerate(testloader):    
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            running_loss += loss.item()
            correct += torch.sum(torch.argmax(outputs,1) == targets).item()
            total += inputs.shape[0]
        test_acc.append(correct/total)
        test_loss.append(running_loss/(idx+1))
  
    np.save('./loss_acc_timing/' + id + '_train_loss.npy', np.asarray(train_loss))
    np.save('./loss_acc_timing/' + id + '_test_loss.npy', np.asarray(test_loss))
    np.save('./loss_acc_timing/' + id + '_train_acc.npy', np.asarray(train_acc))
    np.save('./loss_acc_timing/' + id + '_test_acc.npy', np.asarray(test_acc))


