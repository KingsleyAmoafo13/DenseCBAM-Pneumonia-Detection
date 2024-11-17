import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix, roc_curve, precision_recall_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from model import DenseNet169Custom
from config import Config
import torchvision.transforms as transforms

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test data
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_dataset = ImageFolder(Config.TEST_PATH, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

# Load the trained model
model = DenseNet169Custom(num_classes=2).to(device)
model.load_state_dict(torch.load("logs/best_model_fold_1.pth"))
model.eval()

# Custom threshold for classification
custom_threshold = 0.7  

# Metrics computation
all_preds = []
all_labels = []
all_probs = []  # To store the probabilities for ROC and PR curves

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = torch.softmax(model(images), dim=1)
        probs = outputs[:, 1].cpu().numpy()  # Get probability scores for the positive class
        preds = (probs > custom_threshold).astype(int)
        
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)  # Store probabilities for ROC and PR curves

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

accuracy = accuracy_score(all_labels, all_preds)
auc_score = roc_auc_score(all_labels, all_probs)
precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, average=None)
conf_matrix = confusion_matrix(all_labels, all_preds)


for i in range(len(conf_matrix)):
    tn = np.sum(conf_matrix) - (np.sum(conf_matrix[i, :]) + np.sum(conf_matrix[:, i]) - conf_matrix[i, i])
    fp = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]


print(f"Overall Metrics: Accuracy: {accuracy * 100:.2f}%, AUC: {auc_score:.2f}")
print("Class-based Precision, Recall, F1-Score, and Specificity:")
for i, class_name in enumerate(test_dataset.classes):
    print(f"{class_name:<10} Precision: {precision[i] * 100:.2f}%, Recall: {recall[i] * 100:.2f}%, "
          f"F1-Score: {f1[i] * 100:.2f}%")

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Plot ROC Curve
fpr, tpr, _ = roc_curve(all_labels, all_probs)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Plot Precision-Recall (PR) Curve
precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
pr_auc = auc(recall_curve, precision_curve)
plt.figure(figsize=(10, 6))
plt.plot(recall_curve, precision_curve, label=f'PR Curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall (PR) Curve')
plt.legend(loc='lower left')
plt.grid()
plt.show()
