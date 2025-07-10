import os
import argparse
import numpy as np
from PIL import Image

def process_alpha_channel(input_path, output_path=None):
    """
    Process PNG image to create a black and white image based on alpha channel:
    - Alpha == 0: black pixel
    - Alpha > 0: white pixel
    """
    # Load the image with alpha channel
    img = Image.open(input_path).convert('RGBA')
    
    # Convert to numpy array for processing
    img_array = np.array(img)
    
    # Create a blank black image
    result = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
    
    # Set pixels to white where alpha > 0
    result[img_array[:, :, 3] > 0] = 255
    
    # Convert back to PIL Image
    result_img = Image.fromarray(result, mode='L')
    
    # Determine output path if not provided
    if output_path is None:
        filename, ext = os.path.splitext(input_path)
        output_path = f"{filename}_alpha_mask{ext}"
    
    # Save the result
    result_img.save(output_path)
    print(f"Processed image saved to: {output_path}")
    
    return result_img

def main():
    parser = argparse.ArgumentParser(description="Convert alpha channel to black and white mask")
    parser.add_argument("input_path", help="Path to the input PNG image")
    parser.add_argument("-o", "--output", help="Path to save the output image (optional)")
    
    args = parser.parse_args()
    
    process_alpha_channel(args.input_path, args.output)

if __name__ == "__main__":
    # Hardcoded input path instead of using command line arguments
    input_path = "./test_image/test.png"
    output_path = "alphatobin.png"  # Optional, can be set to None
    
    process_alpha_channel(input_path, output_path)