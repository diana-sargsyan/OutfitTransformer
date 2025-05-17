import time
import torch
from PIL import Image
import clip
import sys
import cv2
import numpy as np
import webcolors
from colorthief import ColorThief
import json
import os
from pathlib import Path


def closest_color_name(hex_value):
    try:
        color_name = webcolors.hex_to_name(hex_value)
    except ValueError:
        rgb_value = webcolors.hex_to_rgb(hex_value)
        color_name = get_closest_color(rgb_value)
    return color_name


def get_closest_color(requested_color):
    min_colors = {}
    for name, hex_code in webcolors.CSS3_NAMES_TO_HEX.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(hex_code)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]


category_mapping = {
    "sleeveless top": "<tops>",
    "t-shirt": "<tops>",
    "long-sleeve t-shirt": "<tops>",
    "jeans": "<bottoms>",
    "shorts": "<bottoms>",
    "skirt": "<bottoms>",
    "mini skirt": "<bottoms>",
    "midi skirt": "<bottoms>",
    "maxi skirt": "<bottoms>",
    "yoga pants": "<bottoms>",
    "leggings": "<bottoms>",
    "joggers": "<bottoms>",
    "cargo pants": "<bottoms>",
    "chinos": "<bottoms>",
    "wide-leg pants": "<bottoms>",
    "culottes": "<bottoms>",
    "palazzo pants": "<bottoms>",
    "capri pants": "<bottoms>",
    "jacket": "<outerwear>",
    "blazer": "<outerwear>",
    "coat": "<outerwear>",
    "trench coat": "<outerwear>",
    "sweater": "<tops>",
    "hoodie": "<outerwear>",
    "cardigan": "<outerwear>",
    "dress": "<all-body>",
    "evening dress": "<all-body>",
    "casual dress": "<all-body>",
    "shirt": "<tops>",
    "blouse": "<tops>",
    "polo shirt": "<tops>",
    "tank top": "<tops>",
    "camisole": "<tops>",
    "sneakers": "<shoes>",
    "boots": "<shoes>",
    "sandals": "<shoes>",
    "flip flops": "<shoes>",
    "loafers": "<shoes>",
    "heels": "<shoes>",
    "ballet flats": "<shoes>",
    "ankle boots": "<shoes>",
    "running shoes": "<shoes>",
    "trainers": "<shoes>",
    "classic shoes": "<shoes>",
    "hat": "<hats>",
    "gloves": "<accessories>",
    "socks": "<accessories>",
    "stockings": "<accessories>",
    "underwear": "<accessories>",
    "bra": "<accessories>",
    "panties": "<accessories>",
    "swimsuit": "<all-body>",
    "bikini": "<all-body>",
    "one-piece swimsuit": "<all-body>",
    "sports bra": "<tops>",
    "compression shirt": "<tops>",
    "necklace": "<jewellery>",
    "bracelet": "<jewellery>",
    "earrings": "<jewellery>",
    "ring": "<jewellery>",
    "watch": "<accessories>",
    "brooch": "<accessories>",
    "bag": "<bags>",
    "backpack": "<bags>",
    "tote bag": "<bags>",
    "crossbody bag": "<bags>",
    "shoulder bag": "<bags>",
    "clutch": "<bags>"
}




def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    detailed_categories = list(category_mapping.keys())
    category_texts = clip.tokenize(detailed_categories).to(device)
    with torch.no_grad():
        category_features = model.encode_text(category_texts)
    return model, preprocess, detailed_categories, category_features, device


def classify_clothing(image, model, preprocess, detailed_categories, category_features, device):
    # Process the preprocessed image
    with torch.no_grad():
        image_features = model.encode_image(image)
    similarities = (image_features @ category_features.T).squeeze(0)
    similarity_scores = similarities.softmax(dim=0)
    top_index = similarity_scores.argmax().item()
    top_detailed_category = detailed_categories[top_index]
    top_score = similarity_scores[top_index].item() * 100
    top_broad_category = category_mapping[top_detailed_category]
    return top_broad_category, top_detailed_category, top_score


def get_dominant_color(image_path):
    color_thief = ColorThief(image_path)
    dominant_color = color_thief.get_color(quality=1)
    hex_value = '#{:02x}{:02x}{:02x}'.format(dominant_color[0], dominant_color[1], dominant_color[2])
    color_name = closest_color_name(hex_value)
    return color_name, hex_value


def apply_bilateral_filter(image):
    return cv2.bilateralFilter(image, 9, 75, 75)


def detect_pattern(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    filtered_image = apply_bilateral_filter(image)
    blurred_image = cv2.GaussianBlur(filtered_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=5)

    stripe_count = 0
    line_angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            line_angles.append(angle)
            stripe_count += 1

    if stripe_count > 10:
        angle_std = np.std(line_angles)
        if angle_std < 15:
            return "striped"
    return "plain"


def generate_descriptor(image_path, top_detailed_category):
    color_name, _ = get_dominant_color(image_path)  # We still get hex_value but don't use it
    pattern = detect_pattern(image_path)
    return f"{color_name} {top_detailed_category} with {pattern} pattern"

def process_images(image_paths, output_dir, model, preprocess, detailed_categories, category_features, device):
    results = {}

    for image_path in image_paths:
        try:
            # Preprocess image
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

            # Classify clothing
            top_broad_category, top_detailed_category, top_score = classify_clothing(
                image, model, preprocess, detailed_categories, category_features, device
            )

            # Generate descriptor
            descriptor = generate_descriptor(image_path, top_detailed_category)

            # Create output data
            output_data = {
                "main_category": top_broad_category,
                "top_category": top_detailed_category,
                "confidence_score": top_score,
                "descriptor": descriptor
            }

            # Generate output filename
            image_name = Path(image_path).stem
            output_path = os.path.join(output_dir, f"{image_name}_analysis.json")

            # Write individual result
            with open(output_path, 'w') as json_file:
                json.dump(output_data, json_file, indent=4)

            results[image_path] = output_data
            print(f"Processed {image_path}")

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            results[image_path] = {"error": str(e)}

    # Write summary results
    summary_path = os.path.join(output_dir, "analysis_summary.json")
    with open(summary_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    return results


def main():
    if len(sys.argv) < 3:
        print("Usage: python classify_clothing.py <input_path> <output_dir>")
        print("input_path can be either a directory or a comma-separated list of image paths")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of image paths
    if ',' in input_path:
        # Handle comma-separated list of images
        image_paths = [path.strip() for path in input_path.split(',')]
    else:
        # Handle directory
        if os.path.isdir(input_path):
            image_extensions = ('.jpg', '.jpeg', '.png', '.webp')
            image_paths = [
                os.path.join(input_path, f)
                for f in os.listdir(input_path)
                if f.lower().endswith(image_extensions)
            ]
        else:
            # Single image file
            image_paths = [input_path]

    # Validate image paths
    image_paths = [path for path in image_paths if os.path.exists(path)]
    if not image_paths:
        print("No valid image paths found!")
        sys.exit(1)

    print(f"Found {len(image_paths)} images to process")

    # Load model once
    print("Loading model...")
    load_start = time.time()
    model, preprocess, detailed_categories, category_features, device = load_model()
    print(f"Model loaded in {time.time() - load_start:.2f} seconds")

    # Process all images
    print("Processing images...")
    process_start = time.time()
    results = process_images(
        image_paths, output_dir, model, preprocess, detailed_categories, category_features, device
    )
    print(f"All images processed in {time.time() - process_start:.2f} seconds")


if __name__ == "__main__":
    main()
