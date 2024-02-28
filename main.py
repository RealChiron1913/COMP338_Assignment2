import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.cm import get_cmap
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

cuda = True  # We don't use GPU for now.
batch_size = 128
log_interval = 100
epochs = 12

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

english_labels = ["T-shirt/top",
                  "Trouser",
                  "Pullover",
                  "Dress",
                  "Coat",
                  "Sandal",
                  "Shirt",
                  "Sneaker",
                  "Bag",
                  "Ankle boot"]


def get_data():
    train_data = datasets.FashionMNIST('data', train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                       ]))
    train_loader = DataLoader(train_data, batch_size=128, shuffle=False, **kwargs)
    return train_loader


def plot_images(train_loader):
    n_samples_seen = 0.
    mean = 0
    std = 0
    for train_batch, train_target in train_loader:
        batch_size = train_batch.shape[0]
        train_batch = train_batch.view(batch_size, -1)
        this_mean = torch.mean(train_batch, dim=1)
        this_std = torch.sqrt(
            torch.mean((train_batch - this_mean[:, None]) ** 2, dim=1))
        mean += torch.sum(this_mean, dim=0)
        std += torch.sum(this_std, dim=0)
        n_samples_seen += batch_size

    mean /= n_samples_seen
    std /= n_samples_seen
    print(mean, std)
    return mean, std


def normalize_data(mean, std):
    train_data = datasets.FashionMNIST('data', train=True, download=False,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=mean.view(1),
                                                                std=std.view(1))]))

    test_data = datasets.FashionMNIST('data', train=False, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=mean.view(1),
                                                               std=std.view(1))]))

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32,
                                              shuffle=False, **kwargs)
    return train_data, test_data, train_loader, test_loader


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=(3, 3), padding=1)
        self.dropout_2d = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(7 * 7 * 20, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.dropout_2d(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = self.dropout_2d(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = x.view(-1, 7 * 7 * 20)  # flatten / reshape
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


def train(model, optimizer, train_loader, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        batch_size = data.shape[0]
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_size

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    train_loss /= len(test_loader.dataset)
    return train_loss


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # sum up batch loss
            _, pred = output.data.max(dim=1)
            # get the index of the max log-probability
            correct += torch.sum(pred == target.data.long()).item()

        test_loss /= len(test_loader.dataset)
        test_accuracy = float(correct) / len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f},'
              ' Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * test_accuracy))
    return test_loss, test_accuracy


def loop_loader(data_loader):
    while True:
        for elem in data_loader:
            yield elem


def find_lr(model, train_loader, init_lr, max_lr, steps, n_batch_per_step=30):
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr)
    current_lr = init_lr
    best_lr = current_lr
    best_loss = float('inf')
    lr_step = (max_lr - init_lr) / steps

    loader = loop_loader(train_loader)
    for i in range(steps):
        mean_loss = 0
        n_seen_samples = 0
        for j, (data, target) in enumerate(loader):
            if j > n_batch_per_step:
                break
            optimizer.zero_grad()
            if cuda:
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            mean_loss += loss.item() * data.shape[0]
            n_seen_samples += data.shape[0]
            optimizer.step()

        mean_loss /= n_seen_samples
        print('Step %i, current LR: %f, loss %f' % (i, current_lr, mean_loss))

        if np.isnan(mean_loss) or mean_loss > best_loss * 4:
            return best_lr / 4

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_lr = current_lr

        current_lr += lr_step
        optimizer.param_groups[0]['lr'] = current_lr

    return best_lr / 4


if __name__ == '__main__':
    train_loader = get_data()
    mean, std = plot_images(train_loader)
    train_data, test_data, train_loader, test_loader = normalize_data(mean, std)
    model = Model()

    if cuda:
        model.cuda()

    model.reset_parameters()
    lr = find_lr(model, train_loader, 1e-4, 1, 100, 30)
    model.reset_parameters()

    print('Best LR', lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max=3,
                                                        last_epoch=-1)

    logs = {'epoch': [], 'train_loss': [], 'test_loss': [],
            'test_accuracy': [], 'lr': []}
    for epoch in range(epochs):
        train_loss = train(model, optimizer, train_loader, epoch)
        test_loss, test_accuracy = test(model, test_loader)
        logs['epoch'].append(epoch)
        logs['train_loss'].append(train_loss)
        logs['test_loss'].append(test_loss)
        logs['test_accuracy'].append(test_accuracy)
        logs['lr'].append(optimizer.param_groups[0]['lr'])
        scheduler.step(epoch)

