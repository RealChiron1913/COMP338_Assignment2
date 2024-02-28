from ast import arg
from torch import nn
from calendar import c
from math import e
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import torchvision
import argparse
from torch.nn.modules.batchnorm import _BatchNorm
from model import SAM, WideResNet, Cutout, EarlyStopping, FashionCNN

 
parser = argparse.ArgumentParser()
parser.add_argument("--adaptive", default=True, type=bool, help="Use adaptive SAM.")
parser.add_argument("--lr", default=0.001, type=float, help="Base learning rate at the start of the training.")
parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
parser.add_argument("--smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
parser.add_argument("--device", default="cuda:0", type=str, help="Device to use for training.")
parser.add_argument("--depth", default=28, type=int, help="Number of layers.")
parser.add_argument("--widen_factor", default=10, type=int, help="How many times wider compared to normal ResNet.")
parser.add_argument("--dropout", default=0.3, type=float, help="Dropout rate.")
parser.add_argument("--threads", default=8, type=int, help="Number of CPU threads for dataloaders.")
parser.add_argument("--mean", default=0.5, type=float, help="Mean of FashionMNIST dataset.")
parser.add_argument("--std", default=0.5, type=float, help="Std of FashionMNIST dataset.")
parser.add_argument("--rho", default=2.0, type=int, help="Rho parameter for SAM.")
parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
parser.add_argument("--model_path", default="model_test.pth", type=str, help="Path to save model.")
parser.add_argument("--num_early_stop", default=5, type=int, help="Number of early stop.")
parser.add_argument("--early_stopping", default=False, type=bool, help="Use early stopping.")
args = parser.parse_args()
criterion = nn.NLLLoss()

def initialize(args, path, seed=42):
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set device
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # Set number of CPU threads for dataloaders
    args.threads = 8 if args.device.type == "cpu" else 16
    torch.set_num_threads(args.threads)
    # Calculate mean and std of FashionMNIST dataset
    args.mean, args.std = calculate_mean_std()
    # Set path to save model
    args.model_path = path
    # Print arguments
    print(args)


def get_train_data():
    # Define train data transformations
    train_transform = transforms.Compose([
        # transforms.RandomCrop(28, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(args.mean, args.std),
        # Cutout(size=12)
    ])

    # Load FashionMNIST train dataset
    trainset = datasets.FashionMNIST('./data', download=True, train=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    return trainloader


def get_test_data():
    # Define test data transformations
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(args.mean, args.std)
    ])

    # Load FashionMNIST test dataset
    testset = datasets.FashionMNIST('./data', download=True, train=False, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    return testloader


def calculate_mean_std():
    # Load FashionMNIST train dataset
    train_set = datasets.FashionMNIST('data', train=True, download=True, transform=transforms.ToTensor())
    # Concatenate all images in the train dataset
    data = torch.cat([d[0] for d in DataLoader(train_set)])
    # Calculate mean and std along each channel
    mean, std = data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])
    print('mean:', mean, 'std:', std)
    return mean, std


def smooth_crossentropy(pred, gold, smoothing=args.smoothing):
    # Calculate smooth cross-entropy loss
    n_class = pred.size(1)
    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    return F.kl_div(pred, one_hot, reduction='batchmean')


def disable_running_stats(model):
    # Disable running stats for batch normalization layers
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    # Enable running stats for batch normalization layers
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


def evaluate(model, dataloader):
    # Evaluate model on the given dataloader
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = model(images)
            # loss = smooth_crossentropy(F.log_softmax(outputs, dim=1), labels).mean()
            loss = F.nll_loss(outputs, labels)
            loss_sum += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    average_loss = loss_sum / len(dataloader)
    return average_loss, accuracy


def new_lr(optimizer, lr):
    # Set new learning rate for optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    

def train(model, optimizer):
    # Get train and test dataloaders
    trainloader, testloader = get_train_data(), get_test_data()
    count = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        loop = tqdm(trainloader, leave=True)
        for images, labels in loop:
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()

            output = model(images)
            loss = F.nll_loss(output, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loop.set_description(f"Epoch [{epoch+1}/{args.epochs}]")
            loop.set_postfix(loss=loss.item(), lr=args.lr)
            
        train_accuracy = correct / total
        train_loss /= len(trainloader)
        print(f"End of Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%")
        torch.save(model, args.model_path)
        test_loss, test_accuracy = evaluate(model, testloader)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%")

        if args.early_stopping:
            early_stopping = EarlyStopping()
            early_stopping(test_loss)
            if early_stopping.early_stop:
                count += 1            
                if count > args.num_early_stop + 3:
                    print('======training over======')
                    break
                elif count > args.num_early_stop:
                    print(f'======Final stable stopping: {count-5} of 3======')
                    early_stopping.counter = 0
                    early_stopping.early_stop = False
                    continue
                else:
                    print(f'======Early stopping: {count} of {args.num_early_stop}======')
                    args.lr = round(args.lr * 0.2, 6)
                    new_lr(optimizer, args.lr)
                    print('learning rate is reduced by 0.2, now is ', args.lr)
                    early_stopping.early_stop = False
                    early_stopping.counter = 0


def test(path):
    # Load model from path and evaluate on test dataset
    testloader = get_test_data()
    model = torch.load(path)
    model.eval()
    test_loss, test_accuracy = evaluate(model, testloader)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%")


def main():
    initialize(args, path='model_ex1_adam.pth')
    model = FashionCNN().to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train(model, optimizer)
    print('======Testing model', args.model_path, '======')
    test(args.model_path)
    

if __name__ == '__main__':
    main()

