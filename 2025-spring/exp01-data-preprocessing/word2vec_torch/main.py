import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
from config import Config
from datamodule import Word2VecDataModule
from typing import Optional
from tqdm import tqdm
import logging
import sys


INFO = logging.info

class SkipGramNegativeSampling(nn.Module):
    """Skip-gram model with negative sampling."""

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)

    def forward(self, target, context, negative_samples) -> torch.Tensor:
        v_target = self.in_embeddings(target)
        v_context = self.out_embeddings(context)
        v_neg = self.out_embeddings(negative_samples)

        positive_score = torch.bmm(v_target.unsqueeze(1), v_context.unsqueeze(2)).squeeze()
        positive_loss = -F.logsigmoid(positive_score)

        negative_score = torch.bmm(v_neg, v_target.unsqueeze(2)).squeeze()
        negative_loss = -F.logsigmoid(-negative_score).sum(axis=1)

        return (positive_loss + negative_loss).mean()


def train_model(model, dataloader, optimizer, device, cfg: Config):
    model.train()
    total_loss = 0
    steps = 0
    iterator = tqdm(dataloader, desc="Train", leave=False, mininterval=cfg.tqdm_mininterval) if cfg.show_batch_progress else dataloader
    if device.type == 'cuda' and cfg.use_amp:
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        for target, context, negative_samples in iterator:
            target, context, negative_samples = target.to(device), context.to(device), negative_samples.to(device)
            optimizer.zero_grad()
            with autocast():
                loss = model(target, context, negative_samples)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            steps += 1
    else:
        for target, context, negative_samples in iterator:
            target, context, negative_samples = target.to(device), context.to(device), negative_samples.to(device)
            optimizer.zero_grad()
            loss = model(target, context, negative_samples)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
    return total_loss / max(steps, 1)


def eval_model(model, dataloader: Optional[torch.utils.data.DataLoader], device, cfg: Config):
    if dataloader is None:
        return None
    model.eval()
    total_loss = 0
    steps = 0
    iterator = tqdm(dataloader, desc="Val", leave=False, mininterval=cfg.tqdm_mininterval) if cfg.show_batch_progress else dataloader
    with torch.no_grad():
        if device.type == 'cuda' and cfg.use_amp:
            from torch.cuda.amp import autocast
            for target, context, negative_samples in iterator:
                target, context, negative_samples = target.to(device), context.to(device), negative_samples.to(device)
                with autocast():
                    loss = model(target, context, negative_samples)
                total_loss += loss.item()
                steps += 1
        else:
            for target, context, negative_samples in iterator:
                target, context, negative_samples = target.to(device), context.to(device), negative_samples.to(device)
                loss = model(target, context, negative_samples)
                total_loss += loss.item()
                steps += 1
    return total_loss / max(steps, 1)


def main(cfg: Config):
    # Configure logging to work with tqdm
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    log.handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler('word2vec.log')]
    # Set format for the handlers
    formatter = logging.Formatter('%(asctime)s - %(module)s - %(levelname)s - %(message)s')
    log.handlers[0].setFormatter(formatter)
    log.handlers[1].setFormatter(formatter)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INFO(f"Device: {device}")

    dm = Word2VecDataModule(cfg)
    dm.prepare_data()
    dm.setup(stage='fit')
    dataloader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    model = SkipGramNegativeSampling(dm.vocab_size, cfg.embedding_dim).to(device)
    if cfg.use_sparse_embeddings:
        optimizer = optim.SparseAdam(model.parameters(), lr=cfg.learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    for epoch in tqdm(range(cfg.num_epochs), desc="Epochs"):
        loss = train_model(model, dataloader, optimizer, device, cfg)
        if val_loader is not None:
            val_loss = eval_model(model, val_loader, device, cfg)
            INFO(f'Epoch {epoch+1}/{cfg.num_epochs}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')
        else:
            INFO(f'Epoch {epoch+1}/{cfg.num_epochs}, Train Loss: {loss:.4f}')

    torch.save(model.state_dict(), cfg.model_state_path)
    INFO(f"Model state saved to {cfg.model_state_path}")
    with open(cfg.vocab_output_path, 'w', encoding='utf-8') as f:
        json.dump(dm.word_to_idx, f, ensure_ascii=False, indent=4)
    INFO(f"Vocabulary saved to {cfg.vocab_output_path}")


if __name__ == "__main__":
    cfg = Config()
    main(cfg)
