"""
PyTorch Dataset and DataLoader for Multi-Modal Video Question Answering
Loads preprocessed visual features, subtitle features, and Q&A data
"""

import os
import torch
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)

class TVQADataset(Dataset):
    """PyTorch Dataset for TVQA multimodal data"""
    
    def __init__(self, 
                 features_dir: str,
                 split: str = 'train',
                 max_qa_per_clip: int = 5,
                 random_qa_sampling: bool = True):
        """
        Initialize TVQA Dataset
        
        Args:
            features_dir: Directory containing preprocessed feature files
            split: Dataset split ('train', 'val', 'test')
            max_qa_per_clip: Maximum Q&A pairs to sample per clip
            random_qa_sampling: Whether to randomly sample Q&A pairs
        """
        self.features_dir = Path(features_dir)
        self.split = split
        self.max_qa_per_clip = max_qa_per_clip
        self.random_qa_sampling = random_qa_sampling
        
        # Load all feature files
        self.feature_files = list(self.features_dir.glob("*_features.pt"))
        self.data_samples = []
        
        # Build dataset samples
        self._build_dataset()
        
        logger.info(f"Loaded {len(self.data_samples)} samples for {split} split")
    
    def _build_dataset(self):
        """Build dataset by loading and organizing all samples"""
        for feature_file in self.feature_files:
            try:
                # Load preprocessed features
                data = torch.load(feature_file, map_location='cpu')
                
                clip_name = data['clip_name']
                visual_features = data['visual_features']
                subtitle_features = data['subtitle_features']
                subtitle_text = data['subtitle_text']
                qa_data = data['qa_data']
                
                # Filter Q&A data by split if needed
                filtered_qa_data = self._filter_qa_by_split(qa_data)
                
                if len(filtered_qa_data) == 0:
                    continue
                
                # Sample Q&A pairs if needed
                if self.max_qa_per_clip > 0 and len(filtered_qa_data) > self.max_qa_per_clip:
                    if self.random_qa_sampling:
                        sampled_qa = random.sample(filtered_qa_data, self.max_qa_per_clip)
                    else:
                        sampled_qa = filtered_qa_data[:self.max_qa_per_clip]
                else:
                    sampled_qa = filtered_qa_data
                
                # Create individual samples for each Q&A pair
                for qa_item in sampled_qa:
                    # Handle test data that may not have answer_idx
                    answer_idx = qa_item.get('answer_idx', -1)  # -1 indicates no ground truth (test data)
                    
                    sample = {
                        'clip_name': clip_name,
                        'visual_features': visual_features,
                        'subtitle_features': subtitle_features,
                        'subtitle_text': subtitle_text,
                        'question': qa_item['q'],
                        'answers': [qa_item['a0'], qa_item['a1'], qa_item['a2'], qa_item['a3'], qa_item['a4']],
                        'correct_answer_idx': answer_idx,
                        'qid': qa_item['qid'],
                        'show_name': qa_item.get('show_name', ''),
                        'timestamp': qa_item.get('ts', ''),
                        'has_answer': answer_idx != -1  # Flag to indicate if ground truth is available
                    }
                    self.data_samples.append(sample)
                    
            except Exception as e:
                logger.warning(f"Error loading {feature_file}: {e}")
    
    def _filter_qa_by_split(self, qa_data: List[Dict]) -> List[Dict]:
        """Filter Q&A data based on split (placeholder - could be enhanced)"""
        # For now, return all Q&A data
        # In practice, you might want to split based on show names, timestamp, or other criteria
        return qa_data
    
    def __len__(self) -> int:
        return len(self.data_samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample"""
        sample = self.data_samples[idx]
        
        return {
            'clip_name': sample['clip_name'],
            'visual_features': sample['visual_features'],  # [30, 2048]
            'subtitle_features': sample['subtitle_features'],  # [768]
            'subtitle_text': sample['subtitle_text'],
            'question': sample['question'],
            'answers': sample['answers'],  # List of 5 answer choices
            'correct_answer_idx': sample['correct_answer_idx'],
            'qid': sample['qid'],
            'show_name': sample['show_name'],
            'timestamp': sample['timestamp'],
            'has_answer': sample['has_answer']  # Flag for ground truth availability
        }

def collate_batch(batch: List[Dict]) -> Dict:
    """Custom collate function for batching"""
    
    # Stack visual features
    visual_features = torch.stack([item['visual_features'] for item in batch])  # [batch_size, 30, 2048]
    
    # Stack subtitle features
    subtitle_features = torch.stack([item['subtitle_features'] for item in batch])  # [batch_size, 768]
    
    # Collect other data
    clip_names = [item['clip_name'] for item in batch]
    subtitle_texts = [item['subtitle_text'] for item in batch]
    questions = [item['question'] for item in batch]
    answers = [item['answers'] for item in batch]  # List of lists
    correct_answer_indices = torch.tensor([item['correct_answer_idx'] for item in batch])
    qids = [item['qid'] for item in batch]
    show_names = [item['show_name'] for item in batch]
    timestamps = [item['timestamp'] for item in batch]
    has_answers = torch.tensor([item['has_answer'] for item in batch])  # Boolean flags
    
    return {
        'clip_names': clip_names,
        'visual_features': visual_features,
        'subtitle_features': subtitle_features,
        'subtitle_texts': subtitle_texts,
        'questions': questions,
        'answers': answers,
        'correct_answer_indices': correct_answer_indices,
        'qids': qids,
        'show_names': show_names,
        'timestamps': timestamps,
        'has_answers': has_answers  # Indicates which samples have ground truth
    }

def create_data_loaders(features_dir: str,
                       batch_size: int = 16,
                       num_workers: int = 4,
                       max_qa_per_clip: int = 5) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders"""
    
    # Create datasets
    train_dataset = TVQADataset(
        features_dir=features_dir,
        split='train',
        max_qa_per_clip=max_qa_per_clip,
        random_qa_sampling=True
    )
    
    val_dataset = TVQADataset(
        features_dir=features_dir,
        split='val',
        max_qa_per_clip=max_qa_per_clip,
        random_qa_sampling=False
    )
    
    test_dataset = TVQADataset(
        features_dir=features_dir,
        split='test',
        max_qa_per_clip=max_qa_per_clip,
        random_qa_sampling=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=True
    )
    
    logger.info(f"Created data loaders:")
    logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    logger.info(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader

# Test the data loader
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    from configs.config import Config
    
    # Test with a small subset
    config = Config()
    features_dir = os.path.join(config.output_dir, 'features')
    
    print("Testing TVQA Dataset...")
    
    # Create a small test dataset
    dataset = TVQADataset(features_dir, split='test', max_qa_per_clip=2)
    
    if len(dataset) > 0:
        print(f"Dataset size: {len(dataset)}")
        
        # Test a single sample
        sample = dataset[0]
        print(f"\nSample structure:")
        print(f"  Clip name: {sample['clip_name']}")
        print(f"  Visual features: {sample['visual_features'].shape}")
        print(f"  Subtitle features: {sample['subtitle_features'].shape}")
        print(f"  Question: {sample['question']}")
        print(f"  Answers: {sample['answers']}")
        print(f"  Correct answer: {sample['correct_answer_idx']}")
        
        # Test data loader
        data_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_batch)
        
        for batch in data_loader:
            print(f"\nBatch structure:")
            print(f"  Visual features: {batch['visual_features'].shape}")
            print(f"  Subtitle features: {batch['subtitle_features'].shape}")
            print(f"  Questions: {len(batch['questions'])}")
            print(f"  Answers: {len(batch['answers'])}")
            break
    else:
        print("No preprocessed features found. Run preprocessing first.")
