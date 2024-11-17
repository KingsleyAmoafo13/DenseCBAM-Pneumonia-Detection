import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder
from model import DenseNet169Custom
from config import Config
from tqdm import tqdm
import random
import numpy as np
from sklearn.model_selection import KFold
import copy

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
def set_seed(seed=Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(Config.SEED)

# Simplified data augmentation
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_val_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
train_dataset = ImageFolder(Config.TRAIN_PATH, transform=transform_train)
test_dataset = ImageFolder(Config.TEST_PATH, transform=transform_val_test)

# Define label smoothing criterion
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        inputs = inputs.to(device)
        targets = targets.to(device)
        confidence = 1.0 - self.smoothing
        log_probs = torch.nn.functional.log_softmax(inputs, dim=-1)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), confidence)
        targets += self.smoothing / (inputs.size(-1) - 1)
        loss = (-targets * log_probs).mean()
        return loss

criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

# Cross-validation setup
kf = KFold(n_splits=2, shuffle=True, random_state=Config.SEED)
train_indices = np.arange(len(train_dataset))

for fold, (train_idx, val_idx) in enumerate(kf.split(train_indices)):
    print(f"\nFold {fold + 1}/{kf.get_n_splits()}")
    train_subsampler = SubsetRandomSampler(train_idx)
    val_subsampler = SubsetRandomSampler(val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, sampler=train_subsampler)
    val_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, sampler=val_subsampler)
    
    # Initialize model, optimizer, and scheduler for each fold
    model = DenseNet169Custom(num_classes=2).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Early stopping criteria
    patience = 5
    best_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None

    for epoch in range(Config.EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{Config.EPOCHS}], Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')

        # Validation evaluation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val
        print(f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            early_stop_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

        scheduler.step(avg_val_loss)

    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), f"logs/best_model_fold_{fold + 1}.pth")
    print("Best model for fold saved.")

torch.save(model.state_dict(), "logs/final_model.pth")
print("Final model saved as final_model.pth.")
