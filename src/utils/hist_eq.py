import os
import json
import cv2
import numpy as np
import csv

# Global dictionary to cache dominant color computations for each image.
dominant_color_dict = {}

def get_dominant_color(img, k=3):
    """
    Uses k-means clustering to find the dominant color in the image.
    
    Args:
        img (np.array): The image read by OpenCV.
        k (int): Number of clusters for k-means.
        
    Returns:
        np.array: The dominant color (in BGR) as a float array.
    """
    pixels = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    label_counts = np.bincount(labels.flatten())
    dominant = centers[np.argmax(label_counts)]
    return dominant

def get_image_dominant_color(path, k=3):
    """
    Reads an image from the given path, computes its dominant color (if not already computed),
    and caches it in the global dictionary.
    
    Args:
        path (str): Path to the image file.
        k (int): Number of clusters for k-means.
        
    Returns:
        np.array or None: The dominant color in BGR, or None if image cannot be read.
    """
    if path in dominant_color_dict:
        return dominant_color_dict[path]
    img = cv2.imread(path)
    if img is None:
        print(f"Warning: Could not read image {path} for dominant color.")
        return None
    dominant = get_dominant_color(img, k)
    dominant_color_dict[path] = dominant
    return dominant

def is_uniform_color_hist(image_paths, correlation_threshold=0.98, bins=8):
    """
    Computes a 3D color histogram for each image and checks if all histograms are highly similar.
    
    Args:
        image_paths (list): List of image file paths.
        correlation_threshold (float): Minimum correlation for two histograms to be considered similar.
        bins (int): Number of bins per color channel.
        
    Returns:
        bool: True if all images have similar color distributions, else False.
    """
    hist_list = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not read image {path}")
            continue
        # Cache dominant color while processing.
        _ = get_image_dominant_color(path, k=3)
        hist = cv2.calcHist([img], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        hist_list.append(hist)
    
    if len(hist_list) == 0:
        return False  # No valid images were read, assume not uniform.
    
    base_hist = hist_list[0]
    for hist in hist_list[1:]:
        corr = cv2.compareHist(base_hist, hist, cv2.HISTCMP_CORREL)
        if corr < correlation_threshold:
            return False
    return True

def process_json_files(json_files, images_folder, correlation_threshold=0.98, bins=8):
    """
    Processes JSON files. For each entry with label 1 in non-test sets, if the images have nearly identical
    color histograms (with a stricter threshold), the set is removed from the JSON.
    
    Args:
        json_files (list): List of JSON file paths.
        images_folder (str): Directory containing the images.
        correlation_threshold (float): Threshold for histogram correlation.
        bins (int): Number of bins per channel for histogram computation.
    """
    updated_data = {}
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Skip processing for the test set.
        if "test" in os.path.basename(json_file).lower():
            print(f"Skipping test set: {json_file}")
            updated_data[json_file] = data
            continue
        
        new_data = []
        for item in data:
            # Process only sets with label 1.
            if item.get("label") == 1:
                image_paths = [os.path.join(images_folder, f"{img_id}.jpg") for img_id in item.get("question", [])]
                if is_uniform_color_hist(image_paths, correlation_threshold, bins):
                    print(f"Removing set with images: {item.get('question')}")
                    continue  # Remove the set by skipping it.
            new_data.append(item)
        updated_data[json_file] = new_data
    
    # Write updated JSON files.
    for json_file, data in updated_data.items():
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Updated JSON saved: {json_file}")
    # Note: No images are removed.

def write_dominant_colors_csv(output_path):
    """
    Writes a CSV file mapping each image (by filename) to its dominant color in RGB format.
    
    Args:
        output_path (str): Output CSV file path.
    """
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image", "dominant_color (R,G,B)"])
        for image_path, color in dominant_color_dict.items():
            # Convert BGR to RGB.
            dominant_color_rgb = [int(color[2]), int(color[1]), int(color[0])]
            writer.writerow([os.path.basename(image_path), dominant_color_rgb])
    print(f"Dominant colors written to: {output_path}")

def main():
    images_folder = "/Users/dinul/Desktop/proj/outfit-transformer/datasets/polyvore/images"  # Folder containing the JPGs.
    json_files = [
        "/Users/dinul/Desktop/proj/outfit-transformer/datasets/polyvore/clean/train_clean.json",
        "/Users/dinul/Desktop/proj/outfit-transformer/datasets/polyvore/clean/valid_clean.json"
    ]
    correlation_threshold = 0.90  # Stricter histogram correlation threshold.
    bins = 8  # Number of bins per channel in histogram.
    
    # Process JSON files (except test) to remove uniform sets.
    process_json_files(json_files, images_folder, correlation_threshold, bins)
    
    # Write out the CSV with dominant color info for all images processed.
    write_dominant_colors_csv("/Users/dinul/Desktop/proj/outfit-transformer/datasets/polyvore/dominant_colors_combo.csv")

if __name__ == "__main__":
    main()
