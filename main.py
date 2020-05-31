from models import *
from gradient import *
from optimizer import *
from dataloading import *
import argparse
import torchvision

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist', type=str, choices=['mnist', 'mnist_small', 'regression'])
parser.add_argument('--optimizer', default='LM', type=str, choices=['LM', 'SGD', 'Adam', 'HF', 'EKFAC','KFAC','lbfgs'])
parser.add_argument('--net_type', default='cnn', type=str, choices=['cnn', 'mlp'])
parser.add_argument('--epoch_num', default=1, type=int)
parser.add_argument('--device', default=2, type=int)
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
    optimizer = LM(model.parameters(), lr=0.5, alpha=1)
elif args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
elif args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
elif args.optimizer == 'lbfgs':
    optimizer  = torch.optim.LBFGS(model.parameters(), lr=0.01,line_search_fn= 'strong_wolfe')
elif args.optimizer == 'HF':
    optimizer = HessianFree(model.parameters(), use_gnm=True, verbose=False)
elif args.optimizer == 'EKFAC':
    # uses SGD or any other optimizer as its base
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    preconditioner = EKFAC(model, 0.1, sua = False,ra=True)
elif args.optimizer == 'KFAC':
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    preconditioner = KFAC(model, 0.1)

if args.dataset != 'regression':
    criterion= nn.CrossEntropyLoss()

id = args.dataset + '_' + args.net_type + '_' + args.optimizer 
print (id)

print ('-------------------training---------------------------')
train_loss = []
test_acc = []
for epoch in range(args.epoch_num):
    for idx, (inputs, targets) in enumerate (trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
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
                    return loss

            record_loss = optimizer.step(closure)
            train_loss.append(record_loss)
        elif args.optimizer == 'HF': 
            def closure():
                z = model(inputs)
                loss = criterion(z, targets)
                loss.backward(create_graph=True)
                return loss, z
            optimizer.zero_grad()
            loss = optimizer.step(closure, M_inv=None)
            train_loss.append(loss)
        elif args.optimizer == 'lbfgs':
            def closure():
                optimizer.zero_grad()
                out = model(inputs)
                loss = criterion(out, targets)
                loss.backward()
                return loss
            loss = optimizer.step(closure)
            train_loss.append(loss)
        elif args.optimizer == 'EKFAC' or args.optimizer == 'KFAC':
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,targets)
            loss.backward()
            preconditioner.step()  # Add a step of preconditioner before the optimizer step.
            optimizer.step()
            train_loss.append(loss)

        elif args.optimizer == 'Adam' or args.optimizer == 'SGD':
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        if idx == 100:
            break

        print ('testing...')
        total, correct = 0, 0
        for idx, (inputs, targets) in enumerate(testloader):    
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            correct += torch.sum(torch.argmax(outputs,1) == targets).item()
            total += inputs.shape[0]
        print (correct/total)
        test_acc.append(correct/total)
  

    np.save('./loss_acc_timing/' + id + '_train_loss.npy', np.asarray(train_loss))
    np.save('./loss_acc_timing/' + id + '_test_acc.npy', np.asarray(test_acc))




