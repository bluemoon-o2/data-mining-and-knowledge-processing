import os
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)
_BASE_DIR = os.path.dirname(__file__)

def get_cache_path(cfg, ds_ref: str, split: str) -> str:
    safe_ref = ds_ref.replace('/', '--')
    p = os.path.join(cfg.cache_dir, safe_ref, f'{split}.csv')
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(_BASE_DIR, p))

def ensure_dir_for_file(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def download_hf_dataset(cfg) -> bool:
    try:
        import datasets as hfds
    except Exception as e:
        logger.warning(f"HuggingFace datasets not available: {e}")
        return False
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    try:
        ds = hfds.load_dataset(cfg.hf_dataset_ref)
    except Exception as e:
        logger.warning(f"Failed to load HF dataset '{cfg.hf_dataset_ref}': {e}")
        return False
    train_df = ds['train'].to_pandas()
    test_df = ds['test'].to_pandas()
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(len(train_df))
    split = int(len(train_df) * (1 - cfg.val_split_ratio))
    tr_idx, dev_idx = idx[:split], idx[split:]
    def map_df(df):
        lab = (df['label'].astype(int) + 1).astype(int)
        title = df['title'].astype(str)
        text = df['content'].astype(str)
        return pd.DataFrame({'label': lab, 'title': title, 'text': text})
    df_tr = map_df(train_df.iloc[tr_idx])
    df_dev = map_df(train_df.iloc[dev_idx])
    df_test = map_df(test_df)
    tp = get_cache_path(cfg, cfg.hf_dataset_ref, 'train')
    vp = get_cache_path(cfg, cfg.hf_dataset_ref, 'val')
    sp = get_cache_path(cfg, cfg.hf_dataset_ref, 'test')
    logger.info(f"HF target paths:\n  train: {tp}\n  val:   {vp}\n  test:  {sp}")
    ensure_dir_for_file(tp)
    ensure_dir_for_file(vp)
    ensure_dir_for_file(sp)
    try:
        df_tr.to_csv(tp, index=False, header=False)
        df_dev.to_csv(vp, index=False, header=False)
        df_test.to_csv(sp, index=False, header=False)
        logger.info("HF dataset CSVs written")
    except Exception as e:
        logger.error(f"Failed to write HF dataset CSVs: {e}")
        return False
    return True


def download_kagglehub_dataset(cfg) -> bool:
    try:
        import kagglehub
    except Exception as e:
        logger.warning(f"KaggleHub not available: {e}")
        return False
    try:
        path = kagglehub.dataset_download(cfg.kagglehub_dataset_ref)
    except Exception as e:
        logger.warning(f"KaggleHub dataset download failed for '{cfg.kagglehub_dataset_ref}': {e}")
        return False
    csv_paths = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith('.csv'):
                csv_paths.append(os.path.join(root, f))
    train_candidates = [p for p in csv_paths if 'train' in os.path.basename(p).lower()]
    test_candidates = [p for p in csv_paths if 'test' in os.path.basename(p).lower()]
    if not train_candidates or not test_candidates:
        logger.warning("KaggleHub download succeeded but train/test CSVs not found; skipping.")
        return False
    try:
        try:
            df_train_raw = pd.read_csv(train_candidates[0], header=None)
            df_test_raw = pd.read_csv(test_candidates[0], header=None)
        except Exception:
            df_train_raw = pd.read_csv(train_candidates[0])
            df_test_raw = pd.read_csv(test_candidates[0])
    except Exception as e:
        logger.warning(f"Failed reading KaggleHub CSVs: {e}")
        return False
    def normalize(df_raw):
        cols = list(df_raw.columns)
        if len(cols) >= 3:
            df = df_raw.iloc[:, :3]
            df.columns = ['label', 'title', 'text']
        elif len(cols) == 2:
            df = df_raw.iloc[:, :2]
            df.columns = ['label', 'text']
            df['title'] = ''
            df = df[['label', 'title', 'text']]
        else:
            return None
        try:
            df['label'] = df['label'].astype(int)
        except Exception:
            df['label'] = 1
        if df['label'].max() > 2:
            df['label'] = df['label'].map({0: 1, 4: 2}).fillna(df['label']).astype(int)
        return df
    df_train = normalize(df_train_raw)
    df_test = normalize(df_test_raw)
    if df_train is None or df_test is None:
        logger.warning("KaggleHub CSV normalization failed: unexpected column layout")
        return False
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(len(df_train))
    split = int(len(df_train) * (1 - cfg.val_split_ratio))
    tr_idx, dev_idx = idx[:split], idx[split:]
    df_tr = df_train.iloc[tr_idx]
    df_dev = df_train.iloc[dev_idx]
    tp = get_cache_path(cfg, cfg.kagglehub_dataset_ref, 'train')
    vp = get_cache_path(cfg, cfg.kagglehub_dataset_ref, 'val')
    sp = get_cache_path(cfg, cfg.kagglehub_dataset_ref, 'test')
    logger.info(f"KaggleHub target paths:\n  train: {tp}\n  val:   {vp}\n  test:  {sp}")
    ensure_dir_for_file(tp)
    ensure_dir_for_file(vp)
    ensure_dir_for_file(sp)
    try:
        df_tr.to_csv(tp, index=False, header=False)
        df_dev.to_csv(vp, index=False, header=False)
        df_test.to_csv(sp, index=False, header=False)
        logger.info("KaggleHub dataset CSVs written")
    except Exception as e:
        logger.error(f"Failed to write KaggleHub dataset CSVs: {e}")
        return False
    return True


def ensure_dataset(cfg) -> bool:
    ref = cfg.kagglehub_dataset_ref if getattr(cfg, 'kagglehub_use', False) else cfg.hf_dataset_ref
    tp = get_cache_path(cfg, ref, 'train')
    vp = get_cache_path(cfg, ref, 'val')
    sp = get_cache_path(cfg, ref, 'test')
    train_exists = os.path.exists(tp)
    val_exists = os.path.exists(vp)
    test_exists = os.path.exists(sp)
    logger.info(f"Checking for cached dataset at: {os.path.dirname(tp)}")
    if train_exists and val_exists and test_exists:
        logger.info("Found cached dataset.")
        return True
    ok = False
    if getattr(cfg, 'kagglehub_use', False):
        logger.info("Attempting KaggleHub dataset download...")
        ok = download_kagglehub_dataset(cfg)
        if ok:
            logger.info("KaggleHub dataset prepared.")
    if not ok:
        logger.info(f"Falling back to HuggingFace dataset '{cfg.hf_dataset_ref}'...")
        ok = download_hf_dataset(cfg)
        if ok:
            logger.info("HuggingFace dataset prepared.")
        else:
            logger.error("Failed to prepare dataset from both Kaggle and HuggingFace.")
    return ok
