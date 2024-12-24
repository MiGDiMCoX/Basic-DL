import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from torch.utils.data import Subset


# Сначала определим на каком устройстве будем работать - GPU или CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 10
batch_size = 100

# Преобразования данных
data_transforms = transforms.Compose([
    transforms.Resize((28, 28)),  # Изменяем размер как для MNIST
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# загрузим отдельно обучающий набор
train_dataset = torchvision.datasets.ImageFolder(root='./animals/train', transform=data_transforms)

# Отбор случайных 10% данных
train_indices = random.sample(range(len(train_dataset)), len(train_dataset) // 10)



# и отдельно загрузим тестовый набор
test_dataset = torchvision.datasets.ImageFolder(root='./animals/val', transform=data_transforms)

test_indices = random.sample(range(len(test_dataset)), len(test_dataset) // 10)


train_subset = Subset(train_dataset, train_indices)
test_subset = Subset(test_dataset, test_indices)


train_loader = torch.utils.data.DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_subset, batch_size=batch_size, shuffle=False)

# сохраним названия этих классов
class_names = train_dataset.classes
num_classes = len(class_names)
print(class_names)


# построим на картинке
inputs, classes = next(iter(train_loader))
img = torchvision.utils.make_grid(inputs, nrow = 5) # метод делает сетку из картинок
img = img.numpy().transpose((1, 2, 0)) # транспонируем для отображения в картинке
plt.imshow(img)
   

# Теперь можно переходить к созданию сети
# Для этого будем использовать как и ранее метод Sequential
# который объединит несколько слоев в один стек
class CnNet(nn.Module):
    def __init__(self, num_classes):
        nn.Module.__init__(self)
        self.layer1 = nn.Sequential(
        # первый сверточный слой с ReLU активацией и maxpooling-ом
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # второй сверточный слой с ReLU активацией и maxpooling-ом
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # классификационный слой
        self.fc = nn.Linear(7*7*32, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1) # флаттеринг
        out = self.fc(out)
        return out




# создаем экземпляр сети
net = CnNet(num_classes).to(device)

# Задаем функцию потерь и алгоритм оптимизации
lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# создаем цикл обучения и замеряем время его выполнения
import time
t = time.time()
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # прямой проход
        outputs = net(images)
        
        # вычисление значения функции потерь
        loss = lossFn(outputs, labels)
        
        # Обратный проход (вычисляем градиенты)
        optimizer.zero_grad()
        loss.backward()
        
        # делаем шаг оптимизации весов
        optimizer.step()
        
        # выводим немного диагностической информации
        if i%100==0:
            print('Эпоха ' + str(epoch) + ' из ' + str(num_epochs) + ' Шаг ' +
                  str(i) + ' Ошибка: ', loss.item())

print('Время: ', time.time() - t)

# посчитаем точность нашей модели: количество правильно классифицированных цифр
# поделенное на общее количество тестовых примеров

correct_predictions = 0
num_test_samples = len(test_dataset)

with torch.no_grad(): # отключим вычисление граиентов, т.к. будем делать только прямой проход
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = net(images) # делаем предсказание по пакету
        _, pred_class = torch.max(pred.data, 1) # выбираем класс с максимальной оценкой
        correct_predictions += (pred_class == labels).sum().item()


print('Точность модели: ' + str(100 * correct_predictions / (num_test_samples / 10)) + '%')

# Нашу модель можно сохранить в файл для дальнейшего использования
torch.save(net.state_dict(), 'CnNet.ckpt')

