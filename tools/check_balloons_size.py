import os
from PIL import Image
import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt

# Path to the generated balloons directory
balloons_dir = '../generated_balloons'

# Check if directory exists
if not os.path.exists(balloons_dir):
    print(f"Directory '{balloons_dir}' does not exist")
    exit(1)

# Get all image files
image_files = [f for f in os.listdir(balloons_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]

if not image_files:
    print(f"No image files found in '{balloons_dir}'")
    exit(0)

# Store image sizes
sizes = []
size_dict = defaultdict(int)

# Process each image
print(f"{'Filename':<30} {'Width':>10} {'Height':>10}")
print("-" * 50)

for filename in sorted(image_files):
    filepath = os.path.join(balloons_dir, filename)
    try:
        with Image.open(filepath) as img:
            width, height = img.size
            sizes.append((width, height))
            size_dict[f"{width}x{height}"] += 1
            print(f"{filename:<30} {width:>10} {height:>10}")
    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("-" * 50)
print(f"Total images: {len(sizes)}")

# Distribution analysis
widths = [s[0] for s in sizes]
heights = [s[1] for s in sizes]

# Create figures for distribution
plt.figure(figsize=(15, 10))

# Width distribution
plt.subplot(2, 2, 1)
plt.hist(widths, bins=20)
plt.title('Width Distribution')
plt.xlabel('Width (pixels)')
plt.ylabel('Frequency')

# Height distribution
plt.subplot(2, 2, 2)
plt.hist(heights, bins=20)
plt.title('Height Distribution')
plt.xlabel('Height (pixels)')
plt.ylabel('Frequency')

# Scatter plot of width vs height
plt.subplot(2, 2, 3)
plt.scatter(widths, heights, alpha=0.5)
plt.title('Width vs Height')
plt.xlabel('Width (pixels)')
plt.ylabel('Height (pixels)')

# Size frequency
plt.subplot(2, 2, 4)
sizes_sorted = sorted(size_dict.items(), key=lambda x: x[1], reverse=True)
labels = [item[0] for item in sizes_sorted[:10]]  # Top 10 most common sizes
values = [item[1] for item in sizes_sorted[:10]]
plt.bar(labels, values)
plt.title('Most Common Sizes')
plt.xlabel('Size (width x height)')
plt.ylabel('Frequency')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('balloon_size_distribution.png')
plt.show()

# Print statistics
print("\nStatistics:")
print(f"Average width: {np.mean(widths):.2f} pixels")
print(f"Average height: {np.mean(heights):.2f} pixels")
print(f"Median width: {np.median(widths):.2f} pixels")
print(f"Median height: {np.median(heights):.2f} pixels")
print(f"Min width: {min(widths)} pixels")
print(f"Min height: {min(heights)} pixels")
print(f"Max width: {max(widths)} pixels")
print(f"Max height: {max(heights)} pixels")

# Print most common sizes
print("\nMost common sizes:")
for size, count in sizes_sorted[:5]:
    print(f"{size}: {count} images")