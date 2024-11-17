import matplotlib.pyplot as plt

# Model names
models = ['Xception', 'VGG16', 'MobileNetV2', 'InceptionV3', 'ResNet50', 'DenseNet169', 
          'ResNet152V2', 'DenseNet121', 'VIT', 'Proposed approach']

# Performance metrics for each model
precision = [0.7971, 0.8126, 0.9003, 0.8897, 0.8233, 0.9133, 0.8702, 0.8927, 0.9245, 0.9556]
recall = [0.7676, 0.8103, 0.9073, 0.8871, 0.8222, 0.9009, 0.8687, 0.8922, 0.9247, 0.9466]
f1_score = [0.7713, 0.8087, 0.9034, 0.8854, 0.8226, 0.9063, 0.8673, 0.8911, 0.9244, 0.95]
accuracy = [0.7676, 0.8103, 0.9087, 0.8871, 0.8222, 0.9135, 0.8687, 0.8922, 0.9247, 0.9535]

# Create a line plot with adjusted y-axis range
plt.figure(figsize=(12, 8))

# Plot each metric with distinct markers
plt.plot(models, precision, marker='o', label='Precision', color='b')
plt.plot(models, recall, marker='s', label='Recall', color='g')
plt.plot(models, f1_score, marker='^', label='F1 Score', color='r')
plt.plot(models, accuracy, marker='d', label='Accuracy', color='purple')

# Adding labels and title
plt.xlabel('Model', fontsize=12)
plt.ylabel('Metric Value (%)', fontsize=12)
plt.ylim(0.75, 0.96)  # Adjusted y-axis range for better differentiation
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Add legend
plt.legend()

# Show the plot
plt.show()
