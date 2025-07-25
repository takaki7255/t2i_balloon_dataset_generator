import os
import glob
from pathlib import Path

def rename_files(directory="./generated_balloons/"):
    """
    Rename all image files in the specified directory to sequential numbers.
    
    Args:
        directory (str): Path to the directory containing images to rename
    """
    # Ensure directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' not found.")
        return
    
    # Get all image files (assuming common image extensions)
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
    
    # Sort files to ensure consistent ordering
    image_files.sort()
    
    # Calculate padding length based on number of files
    padding = len(str(len(image_files)))
    padding = max(3, padding)  # Use at least 3 digits (001, 002, etc.)
    
    # Rename files
    for i, file_path in enumerate(image_files, 1):
        old_path = Path(file_path)
        new_name = f"{i:0{padding}d}{old_path.suffix}"
        new_path = old_path.parent / new_name
        
        os.rename(old_path, new_path)
        print(f"Renamed: {old_path.name} → {new_name}")
    
    print(f"Renamed {len(image_files)} files.")

if __name__ == "__main__":
    rename_files()