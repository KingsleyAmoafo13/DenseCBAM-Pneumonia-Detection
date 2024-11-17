import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# Load the original image from the uploaded file path
image_path = r"C:\Users\MarksLab\Downloads\archive\chest_xray\chest_xray\train\NORMAL\NORMAL2-IM-1316-0001.jpeg"
image = Image.open(image_path)

# Updated augmentation transformations
transform_steps = [
    ("Original Image", transforms.Resize((224, 224))),
    ("Random Horizontal Flip", transforms.RandomHorizontalFlip()),
    ("Random Rotation", transforms.RandomRotation(15)),
    ("Random Resized Crop", transforms.RandomResizedCrop(224, scale=(0.8, 1.0)))
]

# Apply transformations and store the images with titles
transformed_images = [("Original Image", image)]
for title, transform in transform_steps[1:]:  # Skip Resize for original
    transformed_images.append((title, transform(image)))

# Plot the images: Original on top, transformations in a 2x3 grid below
fig, axes = plt.subplots(3, 3, figsize=(10, 10))

# Display the original image on top
axes[0, 1].imshow(transformed_images[0][1], cmap='gray')
axes[0, 1].set_title(transformed_images[0][0])
axes[0, 1].axis("off")

# Display the 3 transformed images in a 2x3 grid below
for idx, (title, img) in enumerate(transformed_images[1:]):
    row, col = divmod(idx, 3)
    axes[row + 1, col].imshow(img, cmap='gray')
    axes[row + 1, col].set_title(title)
    axes[row + 1, col].axis("off")

# Hide any unused subplots
axes[0, 0].axis("off")
axes[0, 2].axis("off")
if len(transformed_images) < 9:
    for i in range(len(transformed_images), 9):
        axes[(i // 3), (i % 3)].axis("off")

plt.tight_layout()
plt.show()
