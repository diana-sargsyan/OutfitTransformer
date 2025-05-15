# import pandas as pd
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from tqdm import tqdm
# import torch

# # Load Mistral model and tokenizer
# model_name = 'mistralai/Mistral-7B-Instruct-v0.2'
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)
# model.eval()  # Set to evaluation mode

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

# # Your predefined strict mapping rules
# mapping_rules = {
#     "sleeveless top": "tops",
#     "t-shirt": "tops",
#     "long-sleeve t-shirt": "tops",
#     "jeans": "bottoms",
#     "shorts": "bottoms",
#     "skirt": "bottoms",
#     "mini skirt": "bottoms",
#     "midi skirt": "bottoms",
#     "maxi skirt": "bottoms",
#     "yoga pants": "bottoms",
#     "leggings": "bottoms",
#     "joggers": "bottoms",
#     "cargo pants": "bottoms",
#     "chinos": "bottoms",
#     "wide-leg pants": "bottoms",
#     "culottes": "bottoms",
#     "palazzo pants": "bottoms",
#     "capri pants": "bottoms",
#     "jacket": "outerwear",
#     "blazer": "outerwear",
#     "coat": "outerwear",
#     "trench coat": "outerwear",
#     "hoodie": "outerwear",
#     "cardigan": "outerwear",
#     "sweater": "tops",
#     "dress": "all-body",
#     "evening dress": "all-body",
#     "casual dress": "all-body",
#     "shirt": "tops",
#     "blouse": "tops",
#     "polo shirt": "tops",
#     "tank top": "tops",
#     "camisole": "tops",
#     "sports bra": "tops",
#     "compression shirt": "tops",
#     "sneakers": "shoes",
#     "boots": "shoes",
#     "sandals": "shoes",
#     "loafers": "shoes",
#     "heels": "shoes",
#     "ballet flats": "shoes",
#     "ankle boots": "shoes",
#     "running shoes": "shoes",
#     "trainers": "shoes",
#     "classic shoes": "shoes",
#     "hat": "hats",
#     "gloves": "accessories",
#     "socks": "accessories",
#     "stockings": "accessories",
#     "underwear": "accessories",
#     "bra": "accessories",
#     "panties": "accessories",
#     "swimsuit": "all-body",
#     "bikini": "all-body",
#     "one-piece swimsuit": "all-body",
#     "necklace": "jewelry",
#     "bracelet": "jewelry",
#     "earrings": "jewelry",
#     "ring": "jewelry",
#     "scarf": "accessories",
#     "watch": "accessories",
#     "brooch": "accessories",
#     "backpack": "bags",
#     "tote bag": "bags",
#     "crossbody bag": "bags",
#     "shoulder bag": "bags",
#     "clutch": "bags"
# }

# # Create a sorted list of unique categories
# categories = sorted(set(mapping_rules.values()))
# categories_str = ", ".join(categories)

# # Function to classify captions using Mistral
# def classify_caption_mistral(caption, categories_str, mapping_rules):
#     # Create a deterministic prompt
#     prompt = (
#         f"Classify the following caption strictly into one of these classes: {categories_str}, "
#         f"or say 'other'.\nCaption: '{caption}'\nAnswer:"
#     )
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
#     # Increase max_new_tokens to allow a full answer and disable gradient computation
#     with torch.no_grad():
#         outputs = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
    
#     # Decode and extract the generated answer
#     generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip().lower()
    
#     # Clean up the output: take the first word in case extra text is generated
#     classification = generated.split()[0]
    
#     return classification if classification in mapping_rules.values() else 'other'

# # Load your CSV
# df = pd.read_csv('/Users/dinul/Desktop/proj/outfit-transformer/captions.csv')

# classified_results = []

# # Process each caption with tqdm
# for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying captions"):
#     caption = row['caption']
#     category = classify_caption_mistral(caption, categories_str, mapping_rules)
#     classified_results.append({
#         'image': row['image'],
#         'caption': caption,
#         'category': category
#     })

# # Save results to CSV
# results_df = pd.DataFrame(classified_results)
# results_df.to_csv('classified_captions.csv', index=False)


import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch

# Use a smaller model for faster inference; distilgpt2 is around 82M parameters.
model_name = 'distilgpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()  # Set the model to evaluation mode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Predefined strict mapping rules
mapping_rules = {
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
    "brooch": "accessories",
    "backpack": "bags",
    "tote bag": "bags",
    "crossbody bag": "bags",
    "shoulder bag": "bags",
    "clutch": "bags"
}

# Create a sorted list of unique categories for prompt clarity
categories = sorted(set(mapping_rules.values()))
categories_str = ", ".join(categories)

# Function to classify captions using the smaller model
def classify_caption(caption, categories_str):
    # Create a deterministic prompt
    prompt = (
        f"Classify the following caption strictly into one of these classes: {categories_str}, "
        f"or say 'other'.\nCaption: '{caption}'\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate a short answer
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
    
    # Decode and extract the answer
    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip().lower()
    classification = generated.split()[0]  # Use the first word as the answer
    
    return classification if classification in mapping_rules.values() else 'other'

# Load your CSV file with captions
df = pd.read_csv('/Users/dinul/Desktop/proj/outfit-transformer/captions.csv')

classified_results = []

# Process each caption with progress tracking
for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying captions"):
    caption = row['caption']
    category = classify_caption(caption, categories_str)
    classified_results.append({
        'image': row['image'],
        'caption': caption,
        'category': category
    })

# Save the results to CSV
results_df = pd.DataFrame(classified_results)
results_df.to_csv('classified_captions.csv', index=False)
