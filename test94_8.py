# 94.8
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sam import SAM
from tqdm import tqdm
import torch.nn.functional as F
import torchvision
import math
import argparse
from torch.nn.modules.batchnorm import _BatchNorm



parser = argparse.ArgumentParser()
parser.add_argument("--adaptive", default=True, type=bool, help="Use adaptive SAM.")
parser.add_argument("--lr", default=0.005, type=float, help="Base learning rate at the start of the training.")
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
args = parser.parse_args()


def smooth_crossentropy(pred, gold, smoothing=args.smoothing):
    n_class = pred.size(1)
    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    return F.kl_div(pred, one_hot, reduction='batchmean')

def initialize(args, seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.threads = 8 if args.device.type == "cpu" else 16
    torch.set_num_threads(args.threads)
    args.mean, args.std = calculate_mean_std()
    # 打印参数
    print(args)


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

# 计算FashionMNIST数据集的均值和标准差
def calculate_mean_std():
    train_set = datasets.FashionMNIST('data', train=True, download=True, transform=transforms.ToTensor())
    data = torch.cat([d[0] for d in DataLoader(train_set)])
    mean, std = data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])
    print('mean:', mean, 'std:', std)
    return mean, std


def get_train_data():
    train_transform = transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(args.mean, args.std),
    ])

    # 加载FashionMNIST数据集
    trainset = datasets.FashionMNIST('./data', download=True, train=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    return trainloader


def get_test_data():
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(args.mean, args.std)
    ])

    # 加载FashionMNIST数据集
    testset = datasets.FashionMNIST('./data', download=True, train=False, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    return testloader

# 定义CNN模型
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(1, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = self._make_layer(block, n, nChannels[0], nChannels[1], 1, dropRate)
        # 2nd block
        self.block2 = self._make_layer(block, n, nChannels[1], nChannels[2], 2, dropRate)
        # 3rd block
        self.block3 = self._make_layer(block, n, nChannels[2], nChannels[3], 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], 10)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def _make_layer(self, block, n, in_planes, out_planes, stride, dropRate):
        layers = []
        for i in range(int(n)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)


    def forward(self, x):
      out = self.conv1(x)
      out = self.block1(out)
      out = self.block2(out)
      out = self.block3(out)
      out = self.relu(self.bn1(out))
      # 使用自适应平均池化以适应不同尺寸的输入
      out = F.adaptive_avg_pool2d(out, (1, 1))
      out = out.view(-1, self.nChannels)
      return self.fc(out)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# 定义评估函数
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = model(images)
            # 使用smooth_crossentropy来计算损失
            loss = smooth_crossentropy(F.log_softmax(outputs, dim=1), labels).mean()
            loss_sum += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    average_loss = loss_sum / len(dataloader)
    return average_loss, accuracy


def new_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def step_lr(epoch):
    if epoch < args.epochs * 3 / 10:
        lr = args.lr
    elif epoch < args.epochs * 6 / 10:
        lr = args.lr * 0.2
    elif epoch < args.epochs * 8 / 10:
        lr = args.lr * 0.2 ** 2
    else:
        lr = args.lr * 0.2 ** 3
    return lr

def train(model, optimizer):
    # 训练模型
    trainloader, testloader = get_train_data(), get_test_data()
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        loop = tqdm(trainloader, leave=True)

        # 更新学习率
        lr = step_lr(epoch)

        # 设置新的学习率
        new_lr(optimizer, lr)

        for images, labels in loop:
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            enable_running_stats(model)
            output = model(images)

            # 使用smooth_crossentropy计算损失
            loss = smooth_crossentropy(F.log_softmax(output, dim=1), labels)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # 第二步的SAM优化
            disable_running_stats(model)
            smooth_crossentropy(F.log_softmax(model(images), dim=1), labels).backward()
            optimizer.second_step(zero_grad=True)

            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_description(f"Epoch [{epoch+1}/{args.epochs}]")
            loop.set_postfix(loss=loss.item(), lr=lr)

        train_accuracy = correct / total
        train_loss /= len(trainloader)
        print(f"End of Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%")
        # 保存模型
        torch.save(model, 'model_test94_8.pth')

        # 测试模型
        test_loss, test_accuracy = evaluate(model, testloader)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%")

        early_stopping(test_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        

def test(path):
    # 测试模型
    testloader = get_test_data()
    model = torch.load(path)
    model.eval()
    test_loss, test_accuracy = evaluate(model, testloader)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%")


def main():
    initialize(args)
    model = WideResNet(depth=args.depth, widen_factor=args.widen_factor, dropRate=args.dropout).to(args.device)

    # 定义损失函数和优化器  
    base_optimizer = torch.optim.SGD  # 定义基础优化器
    optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)  # 使用SAM优化器
    train(model, optimizer)
    test('model_test94_8.pth')
    

if __name__ == '__main__':
    main()