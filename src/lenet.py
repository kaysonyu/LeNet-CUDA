import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import time

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.FashionMNIST('./data', download=True, train=True, transform=transform)
testset = torchvision.datasets.FashionMNIST('./data', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

model = LeNet()
# 推理部分
model.load_state_dict(torch.load("./model/LeNet.pth", weights_only=True))
model = model.to('cuda')

# 训练部分
'''
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)
for epoch in range(80):
    print('epoch ', epoch)
    for inputs, labels in trainloader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
'''

# 推理内存占用测试
'''
images, labels = next(iter(testloader))
images, labels = images.to('cuda'), labels.to('cuda')
initial_mem = torch.cuda.memory_allocated()
model(images)
final_mem = torch.cuda.memory_allocated()
mem_used = final_mem - initial_mem
print(f'Memory used during inference: {mem_used} bytes')
'''

# 参数均方差
'''
def calculate_mse(params):
    mse_total = 0
    for param in params:
        param_flat = param.data.view(-1)
        mean = param_flat.mean()
        mse = ((param_flat - mean) ** 2).mean()
        mse_total += mse
    mse_avg = mse_total / len(params)
    return mse_avg
params = list(model.parameters())
mse = calculate_mse(params)
print(f'Mean Squared Difference of parameters: {mse.item()}')
'''

# 推理时间、准确率测试
correct = 0
total = 0
start_time = time.time()
# initial_mem = torch.cuda.memory_allocated()
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to('cuda'), labels.to('cuda')
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
# final_mem = torch.cuda.memory_allocated()
# mem_used = final_mem - initial_mem
# print(f'Memory used during inference: {mem_used} bytes')
end_time = time.time()
inference_time = end_time - start_time
print(f"Spend: {inference_time}s Accuracy: {correct / total}")

# 训练部分
'''
for name, param in model.named_parameters():
    np.savetxt(f'./params/{name}.txt', param.detach().cpu().numpy().flatten())
torch.save(model.state_dict(), "./model/LeNet.pth")
'''

