import json
import logging
import os
import pathlib
import sys
import tempfile
from argparse import ArgumentParser
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import pickle
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors

import wandb

from ..data import collate_fn
from ..data.datasets import polyvore
from ..evaluation.metrics import compute_cp_scores
from ..models.load import load_model
from ..utils.distributed_utils import cleanup, gather_results, setup
from ..utils.logger import get_logger
from ..utils.loss import FocalLoss
from ..utils.utils import seed_everything

SRC_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
CHECKPOINT_DIR = SRC_DIR / 'checkpoints'
RESULT_DIR = SRC_DIR / 'results'
LOGS_DIR = SRC_DIR / 'logs'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


class ItemDropoutWithReplacement:
    def __init__(self, embedding_dict, metadata, dropout_prob=0.3, top_k=5, cache_path=None):
        """
        Initialize item dropout with similarity-based replacement.
        
        Args:
            embedding_dict (dict): Dictionary mapping item_id to embedding
            metadata (dict): Dictionary containing item metadata with categories
            dropout_prob (float): Probability of replacing an item
            top_k (int): Number of similar items to consider for replacement
            cache_path (str, optional): Path to save/load similarity database
        """
        self.embedding_dict = embedding_dict
        self.metadata = metadata
        self.dropout_prob = dropout_prob
        self.top_k = top_k
        
        # Load or build similarity database
        if cache_path and os.path.exists(cache_path):
            print(f"Loading similarity database from {cache_path}")
            with open(cache_path, 'rb') as f:
                self.similarity_db = pickle.load(f)
        else:
            print("Building similarity database...")
            self.similarity_db = self._build_similarity_database()
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.similarity_db, f)
    
    def _build_similarity_database(self):
        """Build a database of similar items within each category."""
        # Group items by category
        items_by_category = defaultdict(list)
        for item_id, meta in self.metadata.items():
            if item_id in self.embedding_dict and 'category' in meta:
                items_by_category[meta['category']].append(item_id)
        
        similarity_db = {}
        for category, item_ids in tqdm(items_by_category.items(), desc="Processing categories"):
            if len(item_ids) <= self.top_k:
                # Not enough items, use all items in category
                similarity_db[category] = {
                    item_id: [i for i in item_ids if i != item_id]
                    for item_id in item_ids
                }
                continue
            
            # Build feature matrix for nearest neighbor search
            valid_item_ids = []
            features = []
            for item_id in item_ids:
                if item_id in self.embedding_dict:
                    valid_item_ids.append(item_id)
                    features.append(self.embedding_dict[item_id])
            
            if len(valid_item_ids) > self.top_k:
                # Convert to numpy array for nearest neighbor search
                feature_matrix = np.array(features)
                
                # Normalize features
                norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
                feature_matrix = feature_matrix / np.maximum(norms, 1e-10)
                
                # Find nearest neighbors
                nn = NearestNeighbors(n_neighbors=self.top_k+1)
                nn.fit(feature_matrix)
                distances, indices = nn.kneighbors(feature_matrix)
                
                # Store similar items (skip self)
                similarity_db[category] = {}
                for i, item_id in enumerate(valid_item_ids):
                    similar_indices = indices[i, 1:]  # Skip first (self)
                    similar_items = [valid_item_ids[idx] for idx in similar_indices]
                    similarity_db[category][item_id] = similar_items
            else:
                # Not enough valid items
                similarity_db[category] = {
                    item_id: [i for i in valid_item_ids if i != item_id]
                    for item_id in valid_item_ids
                }
        
        return similarity_db
    
    def apply_dropout(self, queries):
        """
        Apply dropout with similarity-based replacement to outfit queries.
        
        Args:
            queries (list): List of queries, each containing item IDs and embeddings
            
        Returns:
            list: Modified queries with some items replaced
        """
        if not isinstance(queries, list):
            # Handle single query case
            return self._dropout_single_query(queries)
        
        return [self._dropout_single_query(query) for query in queries]
    
    def _dropout_single_query(self, query):
        """Apply dropout to a single query."""
        # Extract item IDs and embeddings
        item_ids = query.get('item_ids', [])
        if not item_ids:
            return query  # No items to replace
        
        # Make a copy of the query to modify
        modified_query = {k: v for k, v in query.items()}
        
        # Replace items with probability dropout_prob
        new_item_ids = []
        for item_id in item_ids:
            if np.random.random() < self.dropout_prob and item_id in self.metadata:
                category = self.metadata[item_id].get('category')
                if (category and category in self.similarity_db and 
                    item_id in self.similarity_db[category] and 
                    self.similarity_db[category][item_id]):
                    # Replace with a similar item
                    replacement_id = np.random.choice(self.similarity_db[category][item_id])
                    new_item_ids.append(replacement_id)
                    
                    # Update embeddings if needed
                    if 'embeddings' in modified_query and replacement_id in self.embedding_dict:
                        idx = item_ids.index(item_id)
                        if idx < len(modified_query['embeddings']):
                            modified_query['embeddings'][idx] = self.embedding_dict[replacement_id]
                else:
                    new_item_ids.append(item_id)
            else:
                new_item_ids.append(item_id)
        
        modified_query['item_ids'] = new_item_ids
        return modified_query


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
    parser.add_argument('--n_epochs', type=int,
                        default=200)
    parser.add_argument('--lr', type=float,
                        default=2e-5)
    parser.add_argument('--accumulation_steps', type=int,
                        default=4)
    parser.add_argument('--wandb_key', type=str, 
                        default=None)
    parser.add_argument('--seed', type=int, 
                        default=42)
    parser.add_argument('--checkpoint', type=str, 
                        default=None)
    parser.add_argument('--world_size', type=int, 
                        default=-1)
    parser.add_argument('--project_name', type=str, 
                        default=None)
    parser.add_argument('--demo', action='store_true')
    # Add new argument for item dropout
    parser.add_argument('--use_item_dropout', action='store_true',
                        help='Use item dropout with similarity-based replacement')
    parser.add_argument('--item_dropout_prob', type=float, default=0.3,
                        help='Probability of replacing an item during dropout')
    parser.add_argument('--item_dropout_top_k', type=int, default=5,
                        help='Number of similar items to consider for replacement')
    
    return parser.parse_args()


def setup_dataloaders(rank, world_size, args):
    metadata = polyvore.load_metadata(args.polyvore_dir)
    embedding_dict = polyvore.load_embedding_dict(args.polyvore_dir)
    
    train = polyvore.PolyvoreCompatibilityDataset(
        dataset_dir=args.polyvore_dir, dataset_type=args.polyvore_type, 
        dataset_split='train', metadata=metadata, load_image=False, embedding_dict=embedding_dict
    )
    valid = polyvore.PolyvoreCompatibilityDataset(
        dataset_dir=args.polyvore_dir, dataset_type=args.polyvore_type, 
        dataset_split='valid', metadata=metadata, load_image=False, embedding_dict=embedding_dict
    )
    
    if world_size == 1:
        train_dataloader = DataLoader(
            dataset=train, batch_size=args.batch_sz_per_gpu, shuffle=True,
            num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.cp_collate_fn
        )
        valid_dataloader = DataLoader(
            dataset=valid, batch_size=args.batch_sz_per_gpu, shuffle=False,
            num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.cp_collate_fn
        )
        
    else:
        train_sampler = DistributedSampler(
            train, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        valid_sampler = DistributedSampler(
            valid, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True
        )
        train_dataloader = DataLoader(
            dataset=train, batch_size=args.batch_sz_per_gpu, shuffle=False,
            num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.cp_collate_fn, sampler=train_sampler
        )
        valid_dataloader = DataLoader(
            dataset=valid, batch_size=args.batch_sz_per_gpu, shuffle=False,
            num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.cp_collate_fn, sampler=valid_sampler
        )

    return train_dataloader, valid_dataloader, metadata, embedding_dict


def train_step(
    rank, world_size, 
    args, epoch, logger, wandb_run,
    model, optimizer, scheduler, loss_fn, dataloader,
    item_dropout=None  # New parameter for item dropout
):
    model.train()  
    pbar = tqdm(dataloader, desc=f'Train Epoch {epoch+1}/{args.n_epochs}', disable=(rank != 0))
    
    all_loss, all_preds, all_labels = torch.zeros(1, device=rank), [], []
    for i, data in enumerate(pbar):
        if args.demo and i > 2:
            break
        
        # Apply item dropout if enabled
        if item_dropout is not None and args.use_item_dropout:
            queries = item_dropout.apply_dropout(data['query'])
        else:
            queries = data['query']
            
        labels = torch.tensor(data['label'], dtype=torch.float32).to(rank)
        
        preds = model(queries, use_precomputed_embedding=True).squeeze(1)
        
        loss = loss_fn(y_true=labels, y_prob=preds) / args.accumulation_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if (i + 1) % args.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
        # Accumulate Results
        all_loss += loss.item() * args.accumulation_steps / len(dataloader)
        all_preds.append(preds.detach())
        all_labels.append(labels.detach())

        # Logging 
        score = compute_cp_scores(all_preds[-1], all_labels[-1])
        logs = {
            'loss': loss.item() * args.accumulation_steps,
            'steps': len(pbar) * epoch + i,
            'lr': scheduler.get_last_lr()[0] if scheduler else args.lr,
            **score
        }
        pbar.set_postfix(**logs)
        if args.wandb_key and rank == 0:
            logs = {f'train_{k}': v for k, v in logs.items()}
            wandb_run.log(logs)
    

    all_preds = torch.cat(all_preds).to(rank)
    all_labels = torch.cat(all_labels).to(rank)

    gathered_loss, gathered_preds, gathered_labels = gather_results(all_loss, all_preds, all_labels)
    output = {'loss': gathered_loss.item(), **compute_cp_scores(gathered_preds, gathered_labels)} if rank == 0 else {}
    logger.info(f'Epoch {epoch+1}/{args.n_epochs} --> End {output}')

    return {f'train_{key}': value for key, value in output.items()}
   
        
@torch.no_grad()
def valid_step(
    rank, world_size, 
    args, epoch, logger, wandb_run,
    model, loss_fn, dataloader
):
    model.eval()
    pbar = tqdm(dataloader, desc=f'Valid Epoch {epoch+1}/{args.n_epochs}', disable=(rank != 0))
    
    all_loss, all_preds, all_labels = torch.zeros(1, device=rank), [], []
    for i, data in enumerate(pbar):
        if args.demo and i > 2:
            break
        queries = data['query']
        labels = torch.tensor(data['label'], dtype=torch.float32).to(rank)
    
        preds = model(queries, use_precomputed_embedding=True).squeeze(1)
        
        loss = loss_fn(y_true=labels, y_prob=preds) / args.accumulation_steps
        
        # Accumulate Results
        all_loss += loss.item() * args.accumulation_steps / len(dataloader)
        all_preds.append(preds.detach())
        all_labels.append(labels.detach())

        # Logging
        score = compute_cp_scores(all_preds[-1], all_labels[-1])
        logs = {
            'loss': loss.item() * args.accumulation_steps,
            'steps': len(pbar) * epoch + i,
            **score
        }
        pbar.set_postfix(**logs)
        if args.wandb_key and rank == 0:
            logs = {f'valid_{k}': v for k, v in logs.items()}
            wandb_run.log(logs)
        
    
    all_preds = torch.cat(all_preds).to(rank)
    all_labels = torch.cat(all_labels).to(rank)

    gathered_loss, gathered_preds, gathered_labels = gather_results(all_loss, all_preds, all_labels)
    output = {}
    if rank == 0:
        all_score = compute_cp_scores(gathered_preds, gathered_labels)
        output = {'loss': gathered_loss.item(), **all_score}
        
    logger.info(f'Epoch {epoch+1}/{args.n_epochs} --> End {output}')

    return {f'valid_{key}': value for key, value in output.items()}


def train(
    rank: int, world_size: int, args: Any,
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None
):  
    # Setup
    setup(rank, world_size)
    
    # Logging Setup
    project_name = f'compatibility_{args.model_type}_' + (
        args.project_name if args.project_name 
        else (wandb_run.name if wandb_run else 'test')
    )
    logger = get_logger(project_name, LOGS_DIR, rank)
    logger.info(f'Logger Setup Completed')
    
    # Dataloaders
    train_dataloader, valid_dataloader, metadata, embedding_dict = setup_dataloaders(rank, world_size, args)
    logger.info(f'Dataloaders Setup Completed')
    
    # Initialize item dropout if enabled
    item_dropout = None
    if args.use_item_dropout and rank == 0:
        logger.info(f'Initializing Item Dropout with Similarity-Based Replacement')
        cache_path = os.path.join(args.polyvore_dir, f'similarity_db_{args.item_dropout_top_k}.pkl')
        item_dropout = ItemDropoutWithReplacement(
            embedding_dict=embedding_dict,
            metadata=metadata,
            dropout_prob=args.item_dropout_prob,
            top_k=args.item_dropout_top_k,
            cache_path=cache_path
        )
        logger.info(f'Item Dropout with Similarity-Based Replacement Setup Completed')
    
    # Synchronize item_dropout across processes
    if world_size > 1:
        dist.barrier()
        if rank != 0 and args.use_item_dropout:
            cache_path = os.path.join(args.polyvore_dir, f'similarity_db_{args.item_dropout_top_k}.pkl')
            item_dropout = ItemDropoutWithReplacement(
                embedding_dict=embedding_dict,
                metadata=metadata,
                dropout_prob=args.item_dropout_prob,
                top_k=args.item_dropout_top_k,
                cache_path=cache_path
            )
    
    # Model setting
    model = load_model(model_type=args.model_type, checkpoint=args.checkpoint)
    logger.info(f'Model Loaded and Wrapped with DDP')
    
    # Optimizer, Scheduler, Loss Function
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr, epochs=args.n_epochs, steps_per_epoch=int(len(train_dataloader) / args.accumulation_steps),
        pct_start=0.3, anneal_strategy='cos', div_factor=25, final_div_factor=1e4
    )
    loss_fn = FocalLoss(alpha=0.5, gamma=2) # focal_loss(alpha=0.5, gamma=2)
    logger.info(f'Optimizer and Scheduler Setup Completed')

    # Training Loop
    for epoch in range(args.n_epochs):
        if world_size > 1:
            train_dataloader.sampler.set_epoch(epoch)
        
        # Pass item_dropout to train_step
        train_logs = train_step(
            rank, world_size, 
            args, epoch, logger, wandb_run,
            model, optimizer, scheduler, loss_fn, train_dataloader,
            item_dropout  # Pass item dropout
        )
        
        valid_logs = valid_step(
            rank, world_size, 
            args, epoch, logger, wandb_run,
            model, loss_fn, valid_dataloader
        )
        
        checkpoint_dir = CHECKPOINT_DIR / project_name
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth')
            
        if rank == 0:
            torch.save({
                'config': model.module.cfg.__dict__ if world_size > 1 else model.cfg.__dict__,
                'model': model.state_dict()
            }, checkpoint_path)
            
            score_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}_score.json')
            with open(score_path, 'w') as f:
                score = {**train_logs, **valid_logs}
                json.dump(score, f, indent=4)
            logger.info(f'Checkpoint saved at {checkpoint_path}')
            
        dist.barrier()
        map_location = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
        state_dict = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        model.load_state_dict(state_dict['model'])
        logger.info(f'Checkpoint loaded from {checkpoint_path}')
        
    cleanup()


if __name__ == '__main__':
    args = parse_args()
    
    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        
    if args.wandb_key:
        wandb.login(key=args.wandb_key)
        wandb_run = wandb.init(project='outfit-transformer-cp', config=args.__dict__)
    else:
        wandb_run = None
        
    mp.spawn(
        train, args=(args.world_size, args, wandb_run), 
        nprocs=args.world_size, join=True
    )
