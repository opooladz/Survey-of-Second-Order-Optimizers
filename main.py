from models import *
from gradient import *
from optimizer import *
from dataloading import *
import argparse
import torchvision
from timeit import default_timer as timer
# import lm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist', type=str, choices=['mnist', 'mnist_small', 'regression'])
parser.add_argument('--optimizer', default='LM', type=str, choices=['LM', 'SGD', 'Adam', 'HF', 'EKFAC','KFAC','EKFAC-Adam','KFAC-Adam','lbfgs'])
parser.add_argument('--net_type', default='cnn', type=str, choices=['cnn', 'mlp','mlp_r'])
parser.add_argument('--lr_linesearch', default=False,type=bool)
parser.add_argument('--epoch_num', default=1, type=int)
parser.add_argument('--regression_iters', default=100, type=int)
parser.add_argument('--target_loss',default=0.01,type= float)
parser.add_argument('--target_acc',default=.90,type= float)

parser.add_argument('--device', default=0, type=int)
parser.add_argument('--timer', default=False, type=bool)
args = parser.parse_args()

device = torch.device('cuda:' + str(args.device))
torch.manual_seed(1)
print ('------------------loading data------------------')
if args.dataset == 'mnist':
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_MNIST)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_MNIST)
elif args.dataset == 'mnist_small':
    trainset = MNIST_small(train=True)
    testset = MNIST_small(train=False)
elif args.dataset == 'regression':
    inputs = torch.unsqueeze(torch.linspace(-10, 10, 1000), dim=1).to(device)
    targets = torch.sin(inputs) + 0.2*torch.rand(inputs.size()).to(device)

if args.dataset != 'regression':
<<<<<<< HEAD
    trainloader = DataLoader(trainset, batch_size=600, shuffle=True, num_workers=5)
    testloader = DataLoader(testset, batch_size=600, shuffle=False, num_workers=5)
=======
    trainloader = DataLoader(testset, batch_size=512, shuffle=True, num_workers=5)
    testloader = DataLoader(trainset, batch_size=512, shuffle=False, num_workers=5)
>>>>>>> ecae342beff158632c733fd86929f61b59869a92

print ('------------------initializating network----------------------')

if args.net_type == 'cnn':
    model = CNN().to(device)
elif args.net_type == 'mlp':
    model = MLP().to(device)
elif args.net_type =='mlp_r':
# MLP for regression 
    model = MLP_R().to(device)

if args.optimizer == 'LM':
    optimizer = LM(model.parameters(), lr=1, alpha=1)        
elif args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
elif args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
elif args.optimizer == 'lbfgs':
    optimizer  = torch.optim.LBFGS(model.parameters(), lr=0.01, line_search_fn= 'strong_wolfe')
elif args.optimizer == 'HF':
    optimizer = HessianFree(model.parameters(), use_gnm=True, verbose=False)
elif args.optimizer == 'EKFAC':
    # uses SGD or any other optimizer as its base
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    preconditioner = EKFAC(model, 0.1, sua = False,ra=True)
elif args.optimizer == 'KFAC':
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    preconditioner = KFAC(model, 0.1)
elif args.optimizer == 'EKFAC-Adam':
    # uses Adam or any other optimizer as its base
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    preconditioner = EKFAC(model, 0.1, sua = False,ra=True)
elif args.optimizer == 'KFAC-Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    preconditioner = KFAC(model, 0.1)

if args.dataset != 'regression':
    criterion= nn.CrossEntropyLoss()
else:
    criterion = nn.MSELoss()

id = args.dataset + '_' + args.net_type + '_' + args.optimizer 
print (id)


print ('-------------------training---------------------------')
train_loss = []
test_loss = []
train_acc = []
test_acc = []
ntp = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of Trainable Params: {}'.format(ntp))
dg = torch.tensor([0.01]*ntp,device=device) 
cos = torch.nn.CosineSimilarity(dim=0).to(device)
ls = args.lr_linesearch
target_loss = args.target_loss
target_acc = args.target_acc
time_list = []
if args.dataset != 'regression':
    for epoch in range(args.epoch_num):
        for idx, (inputs, targets) in enumerate (trainloader):
            print ('iter:{}'.format(idx + 1))
            inputs, targets = inputs.to(device), targets.to(device)
            N = inputs.shape[0]
            start = timer()
            if args.optimizer=='LM':
                # if failed iteration run again over same batch
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

                outputs, record_loss, dg = optimizer.step(closure,dg,cos,args.lr_linesearch)
                # dg = optimizer.step(closure,dg,cos,ls)
                correct = torch.sum(torch.argmax(outputs,1) == targets).item()
                train_acc.append(correct/N)
                train_loss.append(record_loss)

            elif args.optimizer == 'HF': 
                def closure():
                    z = model(inputs)
                    loss = criterion(z, targets)
                    loss.backward(create_graph=True)
                    return loss, z
                optimizer.zero_grad()
                loss = optimizer.step(closure, M_inv=None)
                outputs = model(inputs)
                correct = torch.sum(torch.argmax(outputs,1) == targets).item()
                train_acc.append(correct/N)
                train_loss.append(loss)

            elif args.optimizer == 'lbfgs':
                def closure():
                    optimizer.zero_grad()
                    out = model(inputs)
                    loss = criterion(out, targets)
                    loss.backward()
                    return loss
                loss = optimizer.step(closure)
                outputs = model(inputs)
                correct = torch.sum(torch.argmax(outputs,1) == targets).item()
                train_acc.append(correct/N)
                train_loss.append(loss)
            elif args.optimizer == 'EKFAC' or args.optimizer == 'KFAC' or args.optimizer == 'EKFAC-Adam' or args.optimizer == 'KFAC-Adam':
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs,targets)
                loss.backward()
                preconditioner.step()  # Add a step of preconditioner before the optimizer step.
                optimizer.step()
                correct = torch.sum(torch.argmax(outputs,1) == targets).item()
                train_acc.append(correct/N)
                train_loss.append(loss)

            elif args.optimizer == 'Adam' or args.optimizer == 'SGD':
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs,targets)
                loss.backward()
                optimizer.step()
                correct = torch.sum(torch.argmax(outputs,1) == targets).item()
                train_acc.append(correct/N)
                train_loss.append(loss.item())
            end = timer()
            time_list.append(end-start)
            if idx == 100:
                break

            # if args.timer !=True or idx%10 == 0:
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
            if target_acc <= test_acc[-1]: #target_loss >= test_loss[-1] or :
                break


    print(sum(time_list))
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
    print('Final Accuracy: {}'.format(correct/total))

    # np.save('./loss_acc_timing/' + id + '_train_loss_.npy', np.asarray(train_loss))
    # np.save('./loss_acc_timing/' + id + '_test_loss_.npy', np.asarray(test_loss))
    # np.save('./loss_acc_timing/' + id + '_train_acc_.npy', np.asarray(train_acc))
    # np.save('./loss_acc_timing/' + id + '_test_acc_.npy', np.asarray(test_acc))            

else:
    inputs = torch.unsqueeze(torch.linspace(-10, 10, 1000), dim=1).to(device)
    targets = torch.sin(inputs) + 0.2*torch.rand(inputs.size()).to(device)
    if args.optimizer == 'LM':
        for i in range(args.regression_iters):
            def closure(sample=True):
                out = model(inputs)
                diff = (out - targets).squeeze()
                if sample:
                    J = jacobian(diff, model.parameters(), create_graph=True, retain_graph=True).detach()
                    H = J.T @ J
                    g = J.T @ diff
                    loss = torch.mean(diff.detach()**2)
                    return loss, g, H
                else:
                    loss = torch.mean(diff.detach()**2)
                    return out, loss
            optimizer.zero_grad()
            out, loss, dg = optimizer.step(closure, dg_prev=dg, cos=cos)


    elif args.optimizer == 'HF': 
        def closure():
            z = model(inputs)
            loss = criterion(z, targets)
            loss.backward(create_graph=True)
            return loss, z
        for i in range(args.regression_iters):
            optimizer.zero_grad()
            loss = optimizer.step(closure, M_inv=None)
            train_loss.append(loss)
    elif args.optimizer == 'lbfgs':
        for i in range(args.regression_iters):
            def closure():
                optimizer.zero_grad()
                out = model(inputs)
                loss = criterion(out, targets)
                loss.backward()
                return loss
            loss = optimizer.step(closure)
            train_loss.append(loss)
    elif args.optimizer == 'EKFAC' or args.optimizer == 'KFAC' or args.optimizer == 'EKFAC-Adam' or args.optimizer == 'KFAC-Adam':
        for i in range(args.regression_iters):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,targets)
            loss.backward()
            preconditioner.step()  # Add a step of preconditioner before the optimizer step.
            optimizer.step()
            train_loss.append(loss)

    elif args.optimizer == 'Adam' or args.optimizer == 'SGD':
        for i in range(args.regression_iters):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,targets)
            loss.backward()
            optimizer.step()
            correct = torch.sum(torch.argmax(outputs,1) == targets).item()
            train_loss.append(loss.item())

    np.save('./loss_acc_timing/' + id + '_train_loss_sin.npy', np.asarray(train_loss))



'''
    print ('testing...')
    total, correct = 0, 0
    running_loss = 0.0
    for i in range(args.regression_iters):
        inputs, targets = test_inputs.to(device), test_targets.to(device)
        outputs = model(inputs)
        loss = F.mse_loss(outputs, targets)
        running_loss += loss.item()
        correct += torch.sum(torch.argmax(outputs,1) == targets).item()
        total += inputs.shape[0]
    test_acc.append(correct/total)
    test_loss.append(running_loss/(i+1))

    np.save('./loss_acc_timing/' + id + '_train_loss_sin.npy', np.asarray(train_loss))
    np.save('./loss_acc_timing/' + id + '_test_loss_sin.npy', np.asarray(test_loss))
    np.save('./loss_acc_timing/' + id + '_train_acc_sin.npy', np.asarray(train_acc))
    np.save('./loss_acc_timing/' + id + '_test_acc_sin.npy', np.asarray(test_acc))     
'''
