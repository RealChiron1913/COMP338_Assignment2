from pyexpat import model
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import Cutout

def smooth_crossentropy(pred, gold, smoothing=0.1):
    # Calculate smooth cross-entropy loss
    n_class = pred.size(1)
    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    return F.kl_div(pred, one_hot, reduction='batchmean')


def calculate_mean_std():
    # Load FashionMNIST train dataset
    train_set = datasets.FashionMNIST('data', train=True, download=True, transform=transforms.ToTensor())
    # Concatenate all images in the train dataset
    data = torch.cat([d[0] for d in DataLoader(train_set)])
    # Calculate mean and std along each channel
    mean, std = data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])
    return mean, std

def get_test_data():
    # Define test data transformations
    mean, std = calculate_mean_std()
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Load FashionMNIST test dataset
    testset = datasets.FashionMNIST('./data', download=True, train=False, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    return testloader

def evaluate(model, dataloader):
    # Evaluate model on the given dataloader
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            # loss = F.nll_loss(outputs, labels)
            loss = smooth_crossentropy(F.log_softmax(outputs, dim=1), labels).mean()
            loss_sum += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    average_loss = loss_sum / len(dataloader)
    return average_loss, accuracy

def test(path):
    # Load model from path and evaluate on test dataset
    testloader = get_test_data()
    model = torch.load(path)
    model.eval()
    test_loss, test_accuracy = evaluate(model, testloader)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%")


if __name__ == '__main__':
    model_path1 = 'model_ex4_smooth.pth'
    model_path2 = 'model_ex5.pth'

    model1 = torch.load(model_path1)
    model2 = torch.load(model_path2)

    print('Test ' + model_path1)
    test(model_path1)
    print('Test ' + model_path2)
    test(model_path2)
