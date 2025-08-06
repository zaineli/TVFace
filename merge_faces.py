import os
import shutil
from pathlib import Path
from tqdm import tqdm  # For progress bar

def merge_face_images():
    """
    Merge all face images from channel directories into a single 'faces' directory.
    """
    # Define paths
    tvface_dir = os.path.abspath('TVFace')
    output_dir = os.path.join(tvface_dir, 'faces')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # List all directories (channels) in the TVFace directory
    channel_dirs = [d for d in os.listdir(tvface_dir) if os.path.isdir(os.path.join(tvface_dir, d)) 
                   and d != 'faces' and not d.startswith('.')]
    
    print(f"Found {len(channel_dirs)} channel directories")
    
    # Process each channel directory
    total_files = 0
    for channel in channel_dirs:
        channel_path = os.path.join(tvface_dir, channel)
        
        # Find all jpg files in this channel directory
        face_images = [f for f in os.listdir(channel_path) 
                       if f.endswith('.jpg') and os.path.isfile(os.path.join(channel_path, f))]
        
        print(f"Processing {channel}: {len(face_images)} images")
        total_files += len(face_images)
        
        # Copy each image to the faces directory
        for img in tqdm(face_images, desc=f"Copying {channel}"):
            src_path = os.path.join(channel_path, img)
            dst_path = os.path.join(output_dir, img)
            shutil.copy2(src_path, dst_path)  # copy2 preserves metadata
    
    print(f"Merged {total_files} face images into {output_dir}")
    print("Note: Original files in channel directories were preserved.")

if __name__ == "__main__":
    merge_face_images()