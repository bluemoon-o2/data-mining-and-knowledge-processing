import os
import pickle
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
from collections import Counter
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from tqdm import tqdm
from config import Config
from download import ensure_dataset, get_cache_path
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from multiprocessing import Pool, cpu_count
import re
import math


class BatchNegativeSamplerCollate:
    def __init__(self, weights_path: str, num_negative_samples: int, seed: int):
        self.weights_path = weights_path
        self.num_negative_samples = num_negative_samples
        self.seed = seed
        self._weights = None

    def __call__(self, batch):
        targets = torch.tensor([t.item() if isinstance(t, torch.Tensor) else t for t, _ in batch], dtype=torch.long)
        contexts = torch.tensor([c.item() if isinstance(c, torch.Tensor) else c for _, c in batch], dtype=torch.long)
        if self.num_negative_samples <= 0:
            return targets, contexts, torch.empty((len(batch), 0), dtype=torch.long)

        g = torch.Generator()
        g.manual_seed(self.seed)
        if self._weights is None:
            arr = np.memmap(self.weights_path, dtype=np.float32, mode='r')
            self._weights = torch.from_numpy(arr)

        B = len(batch)
        need = torch.full((B,), self.num_negative_samples, dtype=torch.long)
        negatives = torch.empty((B, self.num_negative_samples), dtype=torch.long)
        pos = torch.zeros(B, dtype=torch.long)

        while int(need.sum().item()) > 0:
            draw = torch.multinomial(self._weights, int(need.sum().item()) + 2 * B, replacement=True, generator=g)
            idx = 0
            for b in range(B):
                req = int(need[b].item())
                if req <= 0:
                    continue
                count = 0
                while count < req and idx < draw.numel():
                    k = int(draw[idx].item())
                    idx += 1
                    if k != int(targets[b].item()) and k != int(contexts[b].item()):
                        negatives[b, pos[b].item()] = k
                        pos[b] += 1
                        count += 1
                need[b] = req - count

        return targets, contexts, negatives

logger = logging.getLogger(__name__)
_BASE_DIR = os.path.dirname(__file__)

def _resolve(p: str) -> str:
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(_BASE_DIR, p))
def _build_memmap_indices(corpus_idx: List[List[int]], cache_root: str, split: str):
    idx_path = os.path.join(cache_root, f'{split}.corpus_idx.i32.mm')
    off_path = os.path.join(cache_root, f'{split}.offsets.i64.mm')
    if not (os.path.exists(idx_path) and os.path.exists(off_path)):
        total_len = sum(len(s) for s in corpus_idx)
        idx_mm = np.memmap(idx_path, dtype=np.int32, mode='w+', shape=(total_len,))
        off_mm = np.memmap(off_path, dtype=np.int64, mode='w+', shape=(len(corpus_idx) + 1,))
        pos = 0
        off_mm[0] = 0
        for i, sent in enumerate(corpus_idx):
            n = len(sent)
            if n:
                idx_mm[pos:pos + n] = np.asarray(sent, dtype=np.int32)
            pos += n
            off_mm[i + 1] = pos
        idx_mm.flush()
        off_mm.flush()
    return idx_path, off_path
def _build_memmap_weights(weights: torch.Tensor, cache_root: str):
    w_path = os.path.join(cache_root, 'sampling_weights.f32.mm')
    if not os.path.exists(w_path):
        mm = np.memmap(w_path, dtype=np.float32, mode='w+', shape=(weights.shape[0],))
        mm[:] = weights.cpu().numpy()
        mm.flush()
    return w_path

def _tokenize_text(text, remove_stopwords, min_token_len, stop_words):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    return tokens

def _load_corpus(csv_path: str, cache_file: Optional[str] = None, cfg: Optional[Config] = None) -> List[List[str]]:
    if cache_file and os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            logger.info(f"Loading preprocessed data from cache: {cache_file}")
            return cached.get('corpus', [])
        except Exception as e:
            logger.error(f"Error loading cache file {cache_file}: {e}")

    df = pd.read_csv(csv_path)
    df['text'] = df.iloc[:, 1].astype(str) + " " + df.iloc[:, 2].astype(str)
    
    if cfg is None:
        cfg = Config()

    num_cores = 2
    logger.info(f"Tokenizing corpus using NLTK with {num_cores} cores...")
    
    stop_words = set(stopwords.words('english'))
    
    with Pool(num_cores) as pool:
        tasks = [(text, cfg.remove_stopwords, cfg.min_token_len, stop_words) for text in df['text']]
        tokens = list(tqdm(pool.starmap(_tokenize_text, tasks), total=len(df['text']), desc="Tokenizing corpus"))

    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump({'corpus': tokens}, f)
        logger.info(f"Saved preprocessed data to cache: {cache_file}")
    return tokens


class Word2VecIterableDataset(IterableDataset):
    def __init__(self, indices_path: str, offsets_path: str, window_size: int,
                 seed: int, random_window: bool):
        self.indices_path = indices_path
        self.offsets_path = offsets_path
        self.window_size = window_size
        self.seed = seed
        self.random_window = random_window

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        info = get_worker_info()
        offsets = np.memmap(self.offsets_path, dtype=np.int64, mode='r')
        total_sents = len(offsets) - 1
        if info is None:
            s_start = 0
            s_end = total_sents
        else:
            per_worker = int(math.ceil(total_sents / info.num_workers))
            s_start = info.id * per_worker
            s_end = min(s_start + per_worker, total_sents)
        idx = np.memmap(self.indices_path, dtype=np.int32, mode='r')
        for s in range(s_start, s_end):
            s0 = int(offsets[s])
            s1 = int(offsets[s + 1])
            n = s1 - s0
            for i in range(n):
                tw = int(idx[s0 + i])
                w = self.window_size
                if self.random_window and w > 1:
                    w = int(torch.randint(1, w + 1, (1,), generator=g).item())
                a = max(0, i - w)
                b = min(n, i + w + 1)
                for j in range(a, b):
                    if i == j:
                        continue
                    cw = int(idx[s0 + j])
                    yield (
                        tw,
                        cw,
                    )


class Word2VecDataModule(LightningDataModule):
    """
    Data module for Word2Vec training with negative sampling.

    Args:
        cfg (Config): Configuration object containing hyperparameters and paths.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        self.sampling_weights: Optional[np.ndarray] = None
        self.train_dataset: Optional[IterableDataset] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self._collate_fn = None

    def prepare_data(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        ensure_dataset(self.cfg)

    def setup(self, stage: Optional[str] = None):
        ds_ref = self.cfg.kagglehub_dataset_ref if getattr(self.cfg, 'kagglehub_use', False) else self.cfg.hf_dataset_ref
        train_path = get_cache_path(self.cfg, ds_ref, 'train')
        val_path = get_cache_path(self.cfg, ds_ref, 'val')
        test_path = get_cache_path(self.cfg, ds_ref, 'test')

        safe_ref = ds_ref.replace('/', '--')
        base_cache = self.cfg.cache_dir if os.path.isabs(self.cfg.cache_dir) else os.path.abspath(os.path.join(_BASE_DIR, self.cfg.cache_dir))
        cache_root = os.path.join(base_cache, safe_ref)
        
        sw_suffix = 'sw-removed' if self.cfg.remove_stopwords else 'sw-kept'
        train_cache = os.path.join(cache_root, f'train.nltk.{sw_suffix}.preprocessed.pkl') if self.cfg.cache_dir else None
        val_cache = os.path.join(cache_root, f'val.nltk.{sw_suffix}.preprocessed.pkl') if self.cfg.cache_dir else None
        test_cache = os.path.join(cache_root, f'test.nltk.{sw_suffix}.preprocessed.pkl') if self.cfg.cache_dir else None

        if not self.word_to_idx:
            corpus = _load_corpus(train_path, train_cache, self.cfg)
            all_words = [w for s in corpus for w in s]
            wc = Counter(all_words)
            vocab = [w for w, c in wc.items() if c >= self.cfg.min_count]
            if self.cfg.max_vocab_size is not None:
                vocab = [w for w, _ in sorted(wc.items(), key=lambda x: x[1], reverse=True) if wc[w] >= self.cfg.min_count][: self.cfg.max_vocab_size]
            self.word_to_idx = {w: i for i, w in enumerate(vocab)}
            self.idx_to_word = {i: w for i, w in enumerate(vocab)}
            self.vocab_size = len(vocab)

            corpus_idx = [[self.word_to_idx[w] for w in s if w in self.word_to_idx] for s in corpus]
            counts = Counter(i for sent in corpus_idx for i in sent)
            total = sum(counts.values())
            freqs = np.array([counts.get(i, 0) / total for i in range(self.vocab_size)])
            weights = freqs ** 0.75
            self.sampling_weights = torch.tensor(weights / np.sum(weights), dtype=torch.float)

            if getattr(self.cfg, 'sample', None) and self.cfg.sample > 0:
                keep = (np.sqrt(freqs / self.cfg.sample) + 1.0) * (self.cfg.sample / freqs)
                keep = np.clip(keep, 0.0, 1.0)
                rng = np.random.default_rng(self.cfg.seed)
                subsampled = []
                for sent in corpus_idx:
                    if not sent:
                        continue
                    probs = keep[np.array(sent)]
                    mask = rng.random(len(sent)) < probs
                    filtered = [tok for tok, m in zip(sent, mask) if m]
                    if filtered:
                        subsampled.append(filtered)
                corpus_idx = subsampled

            logger.info(f"Vocabulary size: {self.vocab_size}")
            w_path = _build_memmap_weights(self.sampling_weights, cache_root)
            self._collate_fn = BatchNegativeSamplerCollate(w_path, self.cfg.num_negative_samples, self.cfg.seed)

            t_idx_path, t_off_path = _build_memmap_indices(corpus_idx, cache_root, 'train')
            self.train_dataset = Word2VecIterableDataset(
                t_idx_path,
                t_off_path,
                self.cfg.window_size,
                self.cfg.seed,
                self.cfg.random_window,
            )
            effective_workers = self.cfg.num_workers
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=effective_workers,
                pin_memory=self.cfg.pin_memory,
                drop_last=False,
                persistent_workers=self.cfg.persistent_workers if effective_workers > 0 else False,
                collate_fn=self._collate_fn,
            )

        if stage in (None, 'fit') and val_path and self.val_loader is None:
            val_corpus = _load_corpus(val_path, val_cache, self.cfg)
            val_idx = [[self.word_to_idx[w] for w in s if w in self.word_to_idx] for s in val_corpus]
            v_idx_path, v_off_path = _build_memmap_indices(val_idx, cache_root, 'val')
            self.val_dataset = Word2VecIterableDataset(
                v_idx_path,
                v_off_path,
                self.cfg.window_size,
                self.cfg.seed,
                self.cfg.random_window,
            )
            effective_workers = self.cfg.num_workers
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=effective_workers,
                pin_memory=self.cfg.pin_memory,
                drop_last=False,
                persistent_workers=self.cfg.persistent_workers if effective_workers > 0 else False,
                collate_fn=self._collate_fn,
            )

        if stage in (None, 'test') and test_path and self.test_loader is None:
            test_corpus = _load_corpus(test_path, test_cache, self.cfg)
            test_idx = [[self.word_to_idx[w] for w in s if w in self.word_to_idx] for s in test_corpus]
            x_idx_path, x_off_path = _build_memmap_indices(test_idx, cache_root, 'test')
            self.test_dataset = Word2VecIterableDataset(
                x_idx_path,
                x_off_path,
                self.cfg.window_size,
                self.cfg.seed,
                self.cfg.random_window,
            )
            effective_workers = self.cfg.num_workers
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=effective_workers,
                pin_memory=self.cfg.pin_memory,
                drop_last=False,
                persistent_workers=self.cfg.persistent_workers if effective_workers > 0 else False,
                collate_fn=self._collate_fn,
            )

    def teardown(self, stage: Optional[str] = None):
        if stage in (None, 'fit'):
            self.train_loader = None
            self.val_loader = None
        if stage in (None, 'test'):
            self.test_loader = None

    def predict_dataloader(self) -> Optional[DataLoader]:
        return self.test_loader or self.val_loader

    def train_dataloader(self) -> DataLoader:
        return self.train_loader

    def val_dataloader(self) -> Optional[DataLoader]:
        return self.val_loader

    def test_dataloader(self) -> Optional[DataLoader]:
        return self.test_loader
 
