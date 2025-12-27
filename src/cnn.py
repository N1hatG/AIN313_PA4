import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" {device} is being used")

X_train_tensor = torch.tensor(np.nan_to_num(X_train_g, nan=0.0), dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_g, dtype=torch.long).to(device)

X_test_tensor = torch.tensor(np.nan_to_num(X_test_g, nan=0.0), dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_g, dtype=torch.long).to(device)

# create dataLoader (for batch training)
BATCH_SIZE = 16 # Bellek hatası alırsan düşür
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"data is ready, input size: {X_train_tensor.shape}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActionCNN_Dynamic(nn.Module):
    def __init__(self, input_channels, num_classes, dropout_rate=0.5):
        super(ActionCNN_Dynamic, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(dropout_rate) # Dinamik Dropout
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def run_experiment(lr, batch_size, dropout, epochs=20):
    print(f"\nExperiment: LR={lr}, Batch={batch_size}, Dropout={dropout}")
    
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    test_ds = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    model = ActionCNN_Dynamic(input_channels=75, num_classes=6, dropout_rate=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # training cycle
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    # Test accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    acc = 100 * correct / total
    print(f"Result: %{acc:.2f} Accuracy")
    return acc

experiments = [
    {'name': 'Baseline', 'lr': 0.001, 'batch': 16, 'dropout': 0.5},
    {'name': 'High LR (fast)', 'lr': 0.01,  'batch': 16, 'dropout': 0.5},
    {'name': 'High Batch',      'lr': 0.001, 'batch': 64, 'dropout': 0.5},
    {'name': 'Low Dropout',    'lr': 0.001, 'batch': 16, 'dropout': 0.1},
]

results = []
names = []

for exp in experiments:
    acc = run_experiment(exp['lr'], exp['batch'], exp['dropout'])
    results.append(acc)
    names.append(exp['name'])


plt.figure(figsize=(10, 6))
bars = plt.bar(names, results, color=['blue', 'orange', 'green', 'red'])

plt.ylabel('Accuracy %')
plt.title('CNN Hiperparameter Comparison (Ablation Study)')
plt.ylim(0, 100)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'%{yval:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#best hyperparameters
BEST_LR = 0.001      
BEST_BATCH = 64
BEST_DROPOUT = 0.5   
EPOCHS = 40          


train_ds = TensorDataset(X_train_tensor, y_train_tensor)
test_ds = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_ds, batch_size=BEST_BATCH, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BEST_BATCH, shuffle=False)

model = ActionCNN_Dynamic(input_channels=75, num_classes=6, dropout_rate=BEST_DROPOUT).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=BEST_LR)

train_losses = []
test_accuracies = []

print(f"Final CNN model training is starting (LR={BEST_LR}, Batch={BEST_BATCH})")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Test Performance
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = 100 * correct / total
    train_losses.append(running_loss / len(train_loader))
    test_accuracies.append(acc)
    
 
