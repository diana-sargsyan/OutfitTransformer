import torch
import numpy as np
import clip
from PIL import Image
from sklearn.ensemble import IsolationForest
from collections import defaultdict
import os
from tqdm import tqdm

def isolation_forest_noise_detection(image_paths, category_mapping, output_dir=None):
    """
    Detect non-garment noise using Isolation Forest on CLIP embeddings.
    
    Args:
        image_paths (list): List of paths to fashion item images
        category_mapping (dict): Mapping from subcategories to parent categories
        output_dir (str, optional): Directory to save results
        
    Returns:
        dict: Detection results with image paths as keys and anomaly flags as values
    """
    # Load CLIP model - using the larger ViT-L/14@336px model for better embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    
    print("Extracting CLIP embeddings...")
    # Extract embeddings and tentative categories
    embeddings = []
    tentative_categories = []
    valid_paths = []
    
    for img_path in tqdm(image_paths):
        try:
            # Load and preprocess image
            image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            
            # Get CLIP embedding
            with torch.no_grad():
                image_features = model.encode_image(image)
                
            embeddings.append(image_features.cpu().numpy().flatten())
            
            # Get tentative category through simple classification
            # For simplicity, using categories from the mapping
            categories = list(set(category_mapping.values()))
            text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in categories]).to(device)
            
            with torch.no_grad():
                text_features = model.encode_text(text_inputs)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
            tentative_category = categories[similarity.argmax().item()]
            tentative_categories.append(tentative_category)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    embeddings = np.array(embeddings)
    
    # Compute category statistics for normalization
    category_stats = defaultdict(lambda: {"embeddings": []})
    for i, category in enumerate(tentative_categories):
        category_stats[category]["embeddings"].append(embeddings[i])
    
    # Calculate mean and std for each category
    for category, stats in category_stats.items():
        category_embeddings = np.array(stats["embeddings"])
        stats["mean"] = np.mean(category_embeddings, axis=0)
        stats["std"] = np.std(category_embeddings, axis=0) + 1e-6  # avoid division by zero
    
    # Apply category-aware normalization
    normalized_embeddings = np.zeros_like(embeddings)
    for i, (emb, category) in enumerate(zip(embeddings, tentative_categories)):
        normalized_embeddings[i] = (emb - category_stats[category]["mean"]) / category_stats[category]["std"]
    
    # Train Isolation Forest
    print("Training Isolation Forest...")
    clf = IsolationForest(
        n_estimators=100,
        contamination=0.15,  # estimated noise percentage
        random_state=42,
        n_jobs=-1
    )
    
    # Fit and predict
    y_pred = clf.fit_predict(normalized_embeddings)
    
    # Convert to anomaly scores (negative = anomaly)
    anomaly_scores = clf.decision_function(normalized_embeddings)
    
    # Create results dictionary
    results = {}
    for i, path in enumerate(valid_paths):
        results[path] = {
            "is_anomaly": y_pred[i] == -1,  # -1 for anomalies, 1 for normal
            "anomaly_score": anomaly_scores[i],
            "tentative_category": tentative_categories[i]
        }
    
    # Save results if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories for anomalies and normal items
        anomaly_dir = os.path.join(output_dir, "anomalies")
        normal_dir = os.path.join(output_dir, "normal")
        os.makedirs(anomaly_dir, exist_ok=True)
        os.makedirs(normal_dir, exist_ok=True)
        
        # Save results to respective directories
        for path, result in results.items():
            filename = os.path.basename(path)
            target_dir = anomaly_dir if result["is_anomaly"] else normal_dir
            target_path = os.path.join(target_dir, filename)
            
            try:
                img = Image.open(path)
                img.save(target_path)
            except Exception as e:
                print(f"Error saving {path}: {e}")
    
    return results

# Example usage
if __name__ == "__main__":
    # Category mapping
    category_mapping = {
        "sleeveless top": "tops",
        "t-shirt": "tops",
        "long-sleeve t-shirt": "tops",
        "jeans": "bottoms",
        "shorts": "bottoms",
        "skirt": "bottoms",
        "mini skirt": "bottoms",
        "midi skirt": "bottoms",
        "maxi skirt": "bottoms",
        "yoga pants": "bottoms",
        "leggings": "bottoms",
        "joggers": "bottoms",
        "cargo pants": "bottoms",
        "chinos": "bottoms",
        "wide-leg pants": "bottoms",
        "culottes": "bottoms",
        "palazzo pants": "bottoms",
        "capri pants": "bottoms",
        "jacket": "outerwear",
        "blazer": "outerwear",
        "coat": "outerwear",
        "trench coat": "outerwear",
        "hoodie": "outerwear",
        "cardigan": "outerwear",
        "sweater": "tops",
        "dress": "all-body",
        "evening dress": "all-body",
        "casual dress": "all-body",
        "shirt": "tops",
        "blouse": "tops",
        "polo shirt": "tops",
        "tank top": "tops",
        "camisole": "tops",
        "sports bra": "tops",
        "compression shirt": "tops",
        "sneakers": "shoes",
        "boots": "shoes",
        "sandals": "shoes",
        "loafers": "shoes",
        "heels": "shoes",
        "ballet flats": "shoes",
        "ankle boots": "shoes",
        "running shoes": "shoes",
        "trainers": "shoes",
        "classic shoes": "shoes",
        "hat": "hats",
        "gloves": "accessories",
        "socks": "accessories",
        "stockings": "accessories",
        "underwear": "accessories",
        "bra": "accessories",
        "panties": "accessories",
        "swimsuit": "all-body",
        "bikini": "all-body",
        "one-piece swimsuit": "all-body",
        "necklace": "jewelry",
        "bracelet": "jewelry",
        "earrings": "jewelry",
        "ring": "jewelry",
        "scarf": "accessories",
        "watch": "accessories",
        "brooch": "jewelry",
        "backpack": "bags",
        "tote bag": "bags",
        "crossbody bag": "bags",
        "shoulder bag": "bags",
        "clutch": "bags",
        "hair tie": "accessories",
        "scrunchie": "accessories"
    }
    
    # Sample usage (replace with actual image paths)
    image_paths = ["path/to/images/item1.jpg", "path/to/images/item2.jpg"]
    results = isolation_forest_noise_detection(image_paths, category_mapping, output_dir="noise_detection_results")
    
    # Print results summary
    anomaly_count = sum(1 for r in results.values() if r["is_anomaly"])
    print(f"Found {anomaly_count} anomalies out of {len(results)} images")
