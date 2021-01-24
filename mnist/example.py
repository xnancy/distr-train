from __future__ import print_function
import math 
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import args 

from sklearn.cluster import KMeans 
from torch.utils.data import DataLoader 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def split_dataset_variance(train_dataset): 
    oned_array = [np.reshape(val.numpy(), (784)) for val, target in train_dataset] 
    kmeans = KMeans(n_clusters = 2) 
    kmeans.fit(oned_array)
    cluster1 = [i for i,x in enumerate(kmeans.labels_) if x == 1] 
    cluster2 = [i for i,x in enumerate(kmeans.labels_) if x == 0]
    
    if len(cluster1) < len(cluster2): 
        cluster1 += cluster2[:int((len(cluster1) + len(cluster2))/2 - len(cluster1))]
        cluster2 = cluster2[int((len(cluster1) + len(cluster2))/2 - len(cluster1)):]
    else: 
        cluster2 += cluster1[:int((len(cluster1)+len(cluster2))/2 - len(cluster2))]
        cluster1 = cluster1[int((len(cluster1) + len(cluster2))/2-len(cluster2)):]

    dataset1 = [] 
    dataset1_targets = []
    dataset2 = []
    dataset2_targets = [] 

    for i in cluster1: 
        dataset1.append(np.array(train_dataset[i][0]))
        dataset1_targets.append(train_dataset[i][1])
    for i in cluster2: 
        dataset2.append(np.array(train_dataset[i][0]))
        dataset2_targets.append(train_dataset[i][1])

    return (dataset1, dataset1_targets, dataset2, dataset2_targets)

# run for 1 epoch before merge (50/50)
def train(args, device, model, train_generator, batch_idx,  optimizer, scheduler, steps, epoch):    
    model.train()
    for i in range(steps): 
        (data, target) = next(train_generator)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target.long())
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_generator),
                100. * batch_idx / len(train_generator), loss.item()))
        batch_idx += 1
    return model 
        
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target.long(), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main(): 
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--steps', type=int, default=469, metavar='N',
                        help='number of batch steps to train before merge (default: `10`)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    # Datasets
    dataset_train = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    # Random split 
    # dataset_train1, dataset_train2 = torch.utils.data.random_split(dataset_train, [30000, 30000])
    # Target based split 
    """
    dataset1 = [] 
    dataset1_targets = [] 
    dataset2 = [] 
    dataset2_targets = [] 
    for i in range(0, len(dataset_train)):
        if len(dataset1) == 30000: 
            dataset2.append(np.array(dataset_train[i][0]))
            dataset2_targets.append(dataset_train[i][1])
            continue 
        if len(dataset2) == 30000:
            dataset1.append(np.array(dataset_train[i][0]))
            dataset1_targets.append(dataset_train[i][1])
            continue 
        if dataset_train[i][1] in [1, 4, 7, 9, 10]: 
            dataset1.append(np.array(dataset_train[i][0]))
            dataset1_targets.append(dataset_train[i][1])
        else:
            dataset2.append(np.array(dataset_train[i][0]))
            dataset2_targets.append(dataset_train[i][1])
    """
    dataset1, dataset1_targets, dataset2, dataset2_targets = split_dataset(dataset_train) 

    dataset1_targets = torch.Tensor(dataset1_targets)
    dataset1 = torch.Tensor(dataset1)
    dataset_train1 = TensorDataset(dataset1,dataset1_targets) # create your datset
    dataset_train2 = TensorDataset(torch.Tensor(dataset2), torch.Tensor(dataset2_targets)) 

    dataset_test = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader1 = torch.utils.data.DataLoader(dataset_train1,**train_kwargs)
    train_loader2 = torch.utils.data.DataLoader(dataset_train2,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    model1 = Net().to(device)
    model2 = Net().to(device)
    optimizer1 = optim.Adadelta(model1.parameters(), lr=args.lr)
    optimizer2 = optim.Adadelta(model2.parameters(), lr=args.lr)
    scheduler1 = StepLR(optimizer1, step_size=1, gamma=args.gamma)
    scheduler2 = StepLR(optimizer2, step_size=1, gamma=args.gamma)
   
    for epoch in range(1, args.epochs + 1):
        print("STARTING EPOCH: " + str(epoch))
        train_generator1 = iter(train_loader1)        
        train_generator2 = iter(train_loader2)
        num_batches = len(train_loader1)        
        for i in range(math.ceil(num_batches  / 2 / args.steps)): 
            print("STEP: " + str(i))
            print("MODEL 1")
            model1 = train(args, device, model1, train_generator1, i * args.steps, optimizer1, scheduler2, args.steps, epoch)
            print("MODEL 1 TEST")
            test(model1, device, test_loader)
            print("MODEL 2")
            model2 = train(args, device, model2, train_generator2, i * args.steps, optimizer2, scheduler2, args.steps, epoch)
            print("MODEL 2 TEST") 
            test(model2, device, test_loader) 
            sd1 = model1.state_dict()
            sd2 = model2.state_dict()

            # Average all parameters
            for key in sd1:
                sd2[key] = (sd2[key] + sd1[key]) / 2.

            model1.load_state_dict(sd2)
            model2.load_state_dict(sd2)
            
            print("AVG TEST")
            test(model1, device, test_loader)
        scheduler1.step()
        scheduler2.step()
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == '__main__':
    main()
