# 94.06%
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sam import SAM
from tqdm import tqdm
import torch.nn.functional as F
import torchvision


def smooth_crossentropy(pred, gold, smoothing=0.1):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)

# 设备配置（使用GPU如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set = datasets.FashionMNIST('data', train=True, download=True, transform=transforms.ToTensor())
data = torch.cat([d[0] for d in DataLoader(train_set)])
mean, std = data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])
print(mean, std)


train_transform = transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# 加载FashionMNIST数据集
trainset = datasets.FashionMNIST('./data', download=True, train=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.FashionMNIST('./data', download=True, train=False, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 定义CNN模型
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(-1, 128 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = FashionCNN().to(device)

# 定义损失函数和优化器
criterion = nn.NLLLoss()
base_optimizer = torch.optim.SGD  # 定义基础优化器
optimizer = SAM(model.parameters(), base_optimizer, lr=0.001, momentum=0.9)
# 定义评估函数
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    average_loss = loss_sum / len(dataloader)
    return average_loss, accuracy

# 训练模型
epochs = 200
for epoch in range(epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    loop = tqdm(trainloader, leave=True)  # 添加tqdm进度条
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)

        # 使用smooth_crossentropy计算损失
        loss = smooth_crossentropy(output, labels).mean()
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # 第二步的SAM优化
        smooth_crossentropy(model(images), labels).mean().backward()
        optimizer.second_step(zero_grad=True)

        train_loss += loss.item()  # 直接使用item()
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 更新进度条
        loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
        loop.set_postfix(loss=loss.item())

    # 计算训练集上的平均损失和准确率
    train_accuracy = correct / total
    train_loss /= len(trainloader)
    print(f"End of Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%")

    # 在测试集上评估模型
    test_loss, test_accuracy = evaluate(model, testloader)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%")

