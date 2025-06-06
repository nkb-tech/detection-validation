"""
YOLO to CSV Converter

This script converts YOLO format datasets to CSV format for evaluation.
It processes YOLO format annotations (normalized center_x, center_y, width, height)
and converts them to pixel coordinates (xmin, ymin, xmax, ymax) in CSV format.
"""

import os
from pathlib import Path
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert YOLO format dataset to CSV')
    parser.add_argument('--images_list', type=str, required=True, 
                        help='Path to file with list of image paths, one per line')
    parser.add_argument('--output', type=str, default='output.csv', 
                        help='Path to output CSV file')
    parser.add_argument('--class_names', type=str, default=None,
                        help='Path to file with class names, one per line (optional)')
    return parser.parse_args()

def yolo_to_pixel(box, img_width, img_height):
    """
    Convert YOLO format (center_x, center_y, width, height) to pixel coordinates (xmin, ymin, xmax, ymax).
    
    Args:
        box (list): YOLO format box [center_x, center_y, width, height] normalized to [0, 1]
        img_width (int): Image width in pixels
        img_height (int): Image height in pixels
    
    Returns:
        tuple: Pixel coordinates (xmin, ymin, xmax, ymax)
    """
    center_x, center_y, width, height = box
    
    # Convert from normalized to pixel values
    center_x *= img_width
    width *= img_width
    center_y *= img_height
    height *= img_height
    
    # Calculate bounding box coordinates
    xmin = int(center_x - width / 2)
    ymin = int(center_y - height / 2)
    xmax = int(center_x + width / 2)
    ymax = int(center_y + height / 2)
    
    # Ensure coordinates are within image boundaries
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_width, xmax)
    ymax = min(img_height, ymax)
    
    return xmin, ymin, xmax, ymax

def get_image_dimensions(image_path):
    """
    Get image dimensions (width, height).
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        tuple: (width, height) in pixels, or (None, None) if image cannot be opened
    """
    try:
        with Image.open(image_path) as img:
            return img.width, img.height
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None, None

def get_label_path(image_path):
    """
    Convert image path to corresponding label path.
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        str: Path to the corresponding label file
    """
    img_path_pathlib = Path(image_path)
    lb_path_pathlib = img_path_pathlib.parent.parent / "labels" / (img_path_pathlib.stem + ".txt")
    return str(lb_path_pathlib)

def process_yolo_dataset(image_list_path, class_names=None):
    """
    Process YOLO dataset based on a file with image paths.
    
    Args:
        image_list_path (str): Path to file containing list of image paths
        class_names (list, optional): List of class names
    
    Returns:
        list: List of dictionaries containing processed annotations
    """
    results = []
    
    # Load the list of images
    with open(image_list_path, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]
    
    print(f"Processing {len(image_paths)} images")
    
    # Process each image and its annotation
    for image_path in tqdm(image_paths):
        img_width, img_height = get_image_dimensions(image_path)
        
        if img_width is None or img_height is None:
            continue
        
        # Get the corresponding label file
        label_path = get_label_path(image_path)
        
        # If no annotation file exists, consider the image as having no objects
        if not os.path.exists(label_path):
            continue
        
        # Read annotations
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip().split()
            if len(line) >= 5:
                class_id = int(line[0])
                class_name = class_names[class_id] if class_names and class_id < len(class_names) else str(class_id)
                box = list(map(float, line[1:5]))
                
                # Convert YOLO format to pixel coordinates
                xmin, ymin, xmax, ymax = yolo_to_pixel(box, img_width, img_height)
                
                # Add to results
                results.append({
                    'image_path': image_path,
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                    'label': class_name
                })
    
    return results

def main():
    """Main function to run the YOLO to CSV conversion."""
    args = parse_args()
    
    # Load class names if provided
    class_names = None
    if args.class_names:
        try:
            with open(args.class_names, 'r') as f:
                class_names = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(class_names)} class names")
        except Exception as e:
            print(f"Error loading class names: {e}")
    
    # Process dataset
    results = process_yolo_dataset(args.images_list, class_names)
    
    # Create DataFrame and save to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"Successfully converted dataset to CSV. Saved at {args.output}")
        print(f"Total annotations: {len(df)}")
    else:
        print("No annotations found or processed.")

if __name__ == "__main__":
    main()