import json
import os
import pathlib
from argparse import ArgumentParser
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from src.data import collate_fn
from src.data.datasets import polyvore
from src.evaluation.metrics import compute_cp_scores
from src.models.load import load_model
from src.utils.utils import seed_everything

# Force everything to run on CPU
device = torch.device("cpu")

SRC_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
CHECKPOINT_DIR = SRC_DIR / 'checkpoints'
RESULT_DIR = SRC_DIR / 'results'
LOGS_DIR = SRC_DIR / 'logs'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['original', 'clip'],
                        default='clip')
    parser.add_argument('--polyvore_dir', type=str, 
                        default='./datasets/polyvore')
    parser.add_argument('--polyvore_type', type=str, choices=['nondisjoint', 'disjoint'],
                        default='nondisjoint')
    parser.add_argument('--batch_sz_per_gpu', type=int,
                        default=512)
    parser.add_argument('--n_workers_per_gpu', type=int,
                        default=4)
    parser.add_argument('--wandb_key', type=str, 
                        default=None)
    parser.add_argument('--seed', type=int, 
                        default=42)
    parser.add_argument('--checkpoint', type=str, 
                        default=None)
    parser.add_argument('--demo', action='store_true')
    
    return parser.parse_args()


def validation(args):
    # Helper function to horizontally stack images
    def hstack_images(images):
        valid_images = [im for im in images if im is not None]
        if not valid_images:
            return None
        widths, heights = zip(*(im.size for im in valid_images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in valid_images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
        return new_im

    # Load metadata and embeddings
    metadata = polyvore.load_metadata(args.polyvore_dir)
    embedding_dict = polyvore.load_embedding_dict(args.polyvore_dir)

    test = polyvore.PolyvoreCompatibilityDataset(
        dataset_dir=args.polyvore_dir,
        dataset_type=args.polyvore_type,
        dataset_split='test',
        metadata=metadata,
        embedding_dict=embedding_dict,
        load_image=True  # Enable image loading
    )
    test_dataloader = DataLoader(
        dataset=test, batch_size=args.batch_sz_per_gpu, shuffle=False,
        num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.cp_collate_fn
    )

    # Load and move model to CPU
    model = load_model(model_type=args.model_type, checkpoint=args.checkpoint)
    model = model.to(device)
    model.eval()

    # Directory to save the stacked outfit images
    stacked_save_dir = os.path.join('results', 'classification_and_color_test_outfits')
    os.makedirs(stacked_save_dir, exist_ok=True)

    pbar = tqdm(test_dataloader, desc=f'[Test] Compatibility')
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i, data in enumerate(pbar):
            if args.demo and i > 2:
                break

            # Create labels on the same device as the model
            labels = torch.tensor(data['label'], dtype=torch.float32, device=device)
            preds = model(data['query'], use_precomputed_embedding=True).squeeze(1)

            all_preds.append(preds.detach())
            all_labels.append(labels.detach())

            # For each outfit in the batch, stack images and save them
            for batch_idx, query in enumerate(data['query']):
                images = [item.image for item in query.outfit]
                stacked_img = hstack_images(images)
                if stacked_img is None:
                    continue

                # Get the predicted compatibility score for this outfit
                score = preds[batch_idx].item()

                # Concatenate item identifiers to create a unique filename
                item_ids = [str(item.item_id) for item in query.outfit]
                filename = f"{'_'.join(item_ids)}_{score:.2f}.jpg"

                # Save the horizontally stacked image
                save_path = os.path.join(stacked_save_dir, filename)
                stacked_img.save(save_path)

            score = compute_cp_scores(all_preds[-1], all_labels[-1])
            pbar.set_postfix(**score)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    final_score = compute_cp_scores(all_preds, all_labels)
    print(f"[Test] Compatibility --> {final_score}")

    if args.checkpoint:
        result_dir = os.path.join(
            RESULT_DIR, args.checkpoint.split('/')[-2],
        )
    else:
        result_dir = os.path.join(
            RESULT_DIR, 'compatibility_demo',
        )
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, 'results.json'), 'w') as f:
        json.dump(final_score, f)
    print(f"[Test] Compatibility --> Results saved to {result_dir}")


if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    validation(args)
