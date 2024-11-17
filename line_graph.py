import matplotlib.pyplot as plt

# Epoch data
epochs = [1, 2, 3, 4, 5, 6, 7, 8]

# Training data
training_loss = [0.2276, 0.2070, 0.1960, 0.1955, 0.1925, 0.1868, 0.1893, 0.1859]
training_accuracy = [92.22, 95.59, 97.28, 96.86, 97.47, 97.93, 97.58, 98.04]

# Validation data
validation_loss = [0.1909, 0.1863, 0.1840, 0.1987, 0.1993, 0.1875, 0.1866, 0.1851]
validation_accuracy = [96.89, 97.85, 97.89, 95.78, 95.90, 97.58, 97.32, 97.81]

# Plotting training data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, training_loss, marker='o', label='Training Loss')
plt.plot(epochs, training_accuracy, marker='o', label='Training Accuracy')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.legend()
plt.grid(True)

# Plotting validation data
plt.subplot(1, 2, 2)
plt.plot(epochs, validation_loss, marker='o', label='Validation Loss')
plt.plot(epochs, validation_accuracy, marker='o', label='Validation Accuracy')
plt.title('Validation Loss and Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.legend()
plt.grid(True)

# Show the plots
plt.tight_layout()
plt.show()
