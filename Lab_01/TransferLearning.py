import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
import numpy as np

# 1. Настройка устройства
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Используемое устройство: {device}")

# 2. Подготовка данных (CIFAR-10)
transform = transforms.Compose([
    transforms.Resize((224, 224)), # ResNet ожидает изображения 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Загрузка данных
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Для ускорения работы берем только часть датасета (иначе на Mac обучение займет часы)
train_subset = torch.utils.data.Subset(train_dataset, range(5000))
test_subset = torch.utils.data.Subset(test_dataset, range(1000))

train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=32, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Функция для оценки модели и сбора метрик
def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    return acc, precision, recall, all_labels, all_preds

# ==========================================
# СТРАТЕГИЯ 1: Feature Extraction (Извлечение признаков)
# ==========================================
print("\n--- СТРАТЕГИЯ 1: Feature Extraction ---")
model_fe = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Замораживаем все веса базовой сети
for param in model_fe.parameters():
    param.requires_grad = False

# Заменяем выходной слой (под наши 10 классов)
num_ftrs = model_fe.fc.in_features
model_fe.fc = nn.Linear(num_ftrs, 10)
model_fe = model_fe.to(device)

criterion = nn.CrossEntropyLoss()
# Обучаем ТОЛЬКО новый выходной слой
optimizer_fe = optim.Adam(model_fe.fc.parameters(), lr=0.001)

# Обучение (3 эпохи)
for epoch in range(3):
    model_fe.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_fe.zero_grad()
        outputs = model_fe(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_fe.step()
        running_loss += loss.item()
    print(f"Эпоха {epoch+1}/3, Loss: {running_loss/len(train_loader):.4f}")

# Оценка качества
acc_fe, prec_fe, rec_fe, labels_fe, preds_fe = evaluate_model(model_fe, test_loader)
print(f"Метрики Feature Extraction -> Accuracy: {acc_fe:.4f}, Precision: {prec_fe:.4f}, Recall: {rec_fe:.4f}")

# Построение матрицы ошибок (Confusion Matrix)
cm_fe = confusion_matrix(labels_fe, preds_fe)
plt.figure(figsize=(8, 6))
plt.imshow(cm_fe, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Матрица ошибок - Feature Extraction')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.tight_layout()
plt.ylabel('Истинные метки')
plt.xlabel('Предсказанные метки')
plt.show()

# ==========================================
# СТРАТЕГИЯ 2: Fine-tuning (Дообучение)
# ==========================================
print("\n--- СТРАТЕГИЯ 2: Fine-tuning ---")
model_ft = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Замораживаем все, КРОМЕ последнего блока (layer4) и выходного слоя
for name, param in model_ft.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 10)
model_ft = model_ft.to(device)

# Используем НИЗКИЙ learning rate для дообучения
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.0001)

# Обучение (5 эпох)
for epoch in range(5):
    model_ft.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_ft.zero_grad()
        outputs = model_ft(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_ft.step()
        running_loss += loss.item()
    print(f"Эпоха {epoch+1}/5, Loss: {running_loss/len(train_loader):.4f}")

# Оценка качества
acc_ft, prec_ft, rec_ft, labels_ft, preds_ft = evaluate_model(model_ft, test_loader)
print(f"Метрики Fine-tuning -> Accuracy: {acc_ft:.4f}, Precision: {prec_ft:.4f}, Recall: {rec_ft:.4f}")

# Вывод полного отчета по классам
print("\nОтчет о классификации (Fine-tuning):")
print(classification_report(labels_ft, preds_ft, target_names=classes, zero_division=0))