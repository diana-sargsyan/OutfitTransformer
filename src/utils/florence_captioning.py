
import os
import sys
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from tqdm import tqdm
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Model identifier
model_id = 'microsoft/Florence-2-base'

def load_model_and_processor(model_id: str):
    """Load Florence-2 model and processor for captioning."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        trust_remote_code=True
    ).eval()
    
    processor = AutoProcessor.from_pretrained(
        model_id,
        device_map=device,
        trust_remote_code=True
    )
    return model, processor

def generate_labels(model, processor, task_prompt: str, image: Image.Image, text_input: str = None):
    """Generate caption labels for an image using the Florence-2 model."""
    prompt = task_prompt if text_input is None else task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    output = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    return output

def process_images(image_dir: str, caption_csv: str, model, processor, task_prompt: str):
    captions_data = []

    for img_file in tqdm(os.listdir(image_dir), desc="Processing images"):
        img_path = os.path.join(image_dir, img_file)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error opening {img_path}: {e}", file=sys.stderr)
            continue
        
        caption_output = generate_labels(model, processor, task_prompt, img)
        caption = list(caption_output.values())[0] if isinstance(caption_output, dict) else caption_output
        captions_data.append({"image": img_file, "caption": caption})
    
    # Save the results to a CSV file.
    captions_df = pd.DataFrame(captions_data)
    captions_df.to_csv(caption_csv, index=False)
    print(f"Captions saved to {caption_csv}")
    return captions_df

def main():
    # Directory paths
    image_dir = "/Users/dinul/Desktop/proj/outfit-transformer/datasets/polyvore/images_florence"
    caption_csv = "captions.csv"
    task_prompt = '<CAPTION>'
    model, processor = load_model_and_processor(model_id)
    df = process_images(image_dir, caption_csv, model, processor, task_prompt)

if __name__ == "__main__":
    main()
