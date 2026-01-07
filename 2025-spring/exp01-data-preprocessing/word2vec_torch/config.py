# -*- coding: utf-8 -*-
"""
Configuration file for the Word2Vec model and training process.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional


@dataclass
class Config:
    # --- Word2Vec Model Parameters ---
    embedding_dim: int = 100
    window_size: int = 5
    num_negative_samples: int = 5
    min_count: int = 1  # Words with a frequency below this threshold will be filtered

    # --- Training Parameters ---
    learning_rate: float = 0.001
    batch_size: int = 128
    num_epochs: int = 10
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False
    persistent_workers: bool = True
    seed: int = 42
    show_batch_progress: bool = False
    log_every_n_steps: int = 1000
    tqdm_mininterval: float = 5.0
    random_window: bool = True
    use_sparse_embeddings: bool = True
    sample: float = 1e-3
    use_amp: bool = True
    remove_stopwords: bool = False
    min_token_len: int = 1
    max_vocab_size: Optional[int] = None

    # --- Dataset Configuration ---
    val_split_ratio: float = 0.1
    kagglehub_use: bool = True
    kagglehub_dataset_ref: str = 'kritanjalijain/amazon-reviews'
    hf_dataset_ref: str = 'amazon_polarity'

    # --- Output Files ---
    model_state_path: str = 'word2vec.pt'
    vocab_output_path: str = 'word2vec_vocab.json'
    cache_dir: str = 'word2vec_cache'


    @property
    def as_dict(self) -> Dict:
        return asdict(self)
