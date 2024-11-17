# Re-plotting with labeled dual y-axes including color indicators in the legend
import matplotlib.pyplot as plt

# Epoch data
epochs = [1, 2, 3, 4, 5, 6, 7, 8]

# Training data
training_loss = [0.2276, 0.2070, 0.1960, 0.1955, 0.1925, 0.1868, 0.1893, 0.1859]
training_accuracy = [92.22, 95.59, 97.28, 96.86, 97.47, 97.93, 97.58, 98.04]

# Validation data
validation_loss = [0.1909, 0.1863, 0.1840, 0.1987, 0.1993, 0.1875, 0.1866, 0.1851]
validation_accuracy = [96.89, 97.85, 97.89, 95.78, 95.90, 97.58, 97.32, 97.81]

# Plotting training data with dual y-axes
plt.figure(figsize=(14, 6))

# Training plot
plt.subplot(1, 2, 1)
ax1 = plt.gca()
ax2 = ax1.twinx()
line1, = ax1.plot(epochs, training_loss, marker='o', color='blue', label='Training Loss')
line2, = ax2.plot(epochs, training_accuracy, marker='o', color='orange', label='Training Accuracy')
ax1.set_ylim(0.18, 0.24)  # Adjusted range for training loss
ax2.set_ylim(90, 100)  # Adjusted range for training accuracy
ax1.set_title('Training Loss and Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color='blue')
ax2.set_ylabel('Accuracy', color='orange')
ax1.grid(True)
ax1.legend(handles=[line1, line2], loc='upper left')

# Validation plot
plt.subplot(1, 2, 2)
ax1 = plt.gca()
ax2 = ax1.twinx()
line3, = ax1.plot(epochs, validation_loss, marker='o', color='blue', label='Validation Loss')
line4, = ax2.plot(epochs, validation_accuracy, marker='o', color='orange', label='Validation Accuracy')
ax1.set_ylim(0.18, 0.24)  # Adjusted range for validation loss
ax2.set_ylim(90, 100)  # Adjusted range for validation accuracy
ax1.set_title('Validation Loss and Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color='blue')
ax2.set_ylabel('Accuracy', color='orange')
ax1.grid(True)
ax1.legend(handles=[line3, line4], loc='upper left')

# Show the plots
plt.tight_layout()
plt.show()
