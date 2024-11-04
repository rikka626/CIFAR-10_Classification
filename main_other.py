import torch
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from ResNet import ResNet18
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm  # 引入 tqdm

############################ 超参数区域 ############################
batch_size = 4
epochs = 15
learning_rate = 2e-3
momentum = 0.9
step_size = 10
gamma = 0.5
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

############################ 设备配置 ############################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

############################ 训练函数 ############################
def train_one_epoch(model, trainloader, criterion, optimizer):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    correct = 0
    total = 0

    scaler = torch.cuda.amp.GradScaler()  # 创建 GradScaler 实例
    # 添加进度条
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader), desc="Training"):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # 混合精度上下文
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # 反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


    accuracy = 100 * correct / total
    return running_loss / len(trainloader), accuracy

############################ 验证函数 ############################
def validate_model(model, testloader, criterion):
    model.eval()  # 设置模型为评估模式
    running_loss = 0.0  # 初始化损失累加器
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)  # 计算当前批次的损失
            running_loss += loss.item()  # 累加损失

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(testloader)  # 计算平均损失
    return avg_loss, accuracy, all_labels, all_preds


############################ 数据可视化 ############################
def plot_confusion_matrix(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

def plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy', color='green')
    plt.plot(val_accuracies, label='Validation Accuracy', color='red')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

############################ 主循环 ############################
def main():
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 训练集
    trainset = tv.datasets.CIFAR10(
        root='Dataset/',
        train=True,
        download=True,
        transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True)

    # 测试集
    testset = tv.datasets.CIFAR10(
        root='Dataset/',
        train=False,
        download=True,
        transform=transform)

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=2,
        shuffle=False)


    n_class = 10
    model = ResNet18()
    model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = nn.Linear(512, n_class)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # 添加学习率调度器
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    best_accuracy = 0.0  # 初始化最佳准确率

    # 主循环训练过程
    for epoch in range(epochs):
        train_loss, train_accuracy = train_one_epoch(model, trainloader, criterion, optimizer)
        train_losses.append(train_loss)  # 记录训练损失
        train_accuracies.append(train_accuracy)  # 记录训练准确率


        # 在每个epoch后验证模型
        val_loss, val_accuracy, all_labels, all_preds = validate_model(model, testloader, criterion)
        val_losses.append(val_loss)  # 记录验证损失
        val_accuracies.append(val_accuracy)  # 记录验证准确率

        # print(f'Epoch [{epoch + 1}/{epochs}]')
        # print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}')
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # 如果当前验证准确率高于最佳准确率，则保存模型权重
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), './best_model.pth')  # 保存模型权重

        # 更新学习率
        scheduler.step(val_loss)

    # 绘制混淆矩阵
    plot_confusion_matrix(all_labels, all_preds)
    # 绘制损失和准确率曲线
    plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies)

if __name__ == "__main__":
    main()
