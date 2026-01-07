import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
import re
import pickle
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from word2vec_torch.download import ensure_dataset, get_cache_path
from word2vec_torch.config import Config as TorchConfig

from nltk.corpus import stopwords

def preprocess_text(text, remove_stopwords_flag=False, min_token_len=1):
    """文本预处理函数"""
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    
    if remove_stopwords_flag:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words and len(token) >= min_token_len and token.isalpha()]
    else:
        tokens = [token for token in tokens if len(token) >= min_token_len and token.isalpha()]
    return tokens

def load_and_preprocess_data(file_path, cache_path: str, cfg: TorchConfig):
    """加载并预处理数据，支持缓存读写和多核处理"""
    if cache_path and os.path.exists(cache_path):
        print(f"Loading preprocessed data from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        return data['corpus'], data['labels']

    print(f"Loading raw data from: {file_path}")
    df = pd.read_csv(file_path)
    df['text'] = df.iloc[:, 1].astype(str) + " " + df.iloc[:, 2].astype(str)

    num_cores = 2
    print(f"Preprocessing text data using {num_cores} cores...")
    
    from functools import partial
    preprocess_func = partial(preprocess_text, 
                              remove_stopwords_flag=cfg.remove_stopwords, 
                              min_token_len=cfg.min_token_len)

    with Pool(num_cores) as pool:
        corpus = list(tqdm(pool.imap(preprocess_func, df['text']), total=len(df), desc="Preprocessing text"))

    labels = df.iloc[:, 0].values

    if cache_path:
        print(f"Saving preprocessed data to cache: {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({'corpus': corpus, 'labels': labels}, f)

    return corpus, labels

def train_word2vec(corpus):
    """训练Word2Vec模型"""
    print("Training Word2Vec model...")
    model = Word2Vec(
        sentences=corpus,
        vector_size=100,  # 词向量维度
        window=5,         # 上下文窗口大小
        min_count=1,      # 词频阈值
        workers=4,        # 训练的线程数
    )
    print("Model training complete.")
    return model

def get_document_vector(tokens, model):
    """获取文档的词向量表示（取平均）"""
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(model.vector_size)

def train():
    # --- Setup: NLTK and Dataset Cache ---
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK tokenizers and stopwords...")
        nltk.download('punkt')
        nltk.download('stopwords')

    cfg = TorchConfig()
    ensure_dataset(cfg)
    
    ds_ref = cfg.kagglehub_dataset_ref if getattr(cfg, 'kagglehub_use', False) else cfg.hf_dataset_ref
    data_path = get_cache_path(cfg, ds_ref, 'train')

    # --- Construct Cache Path for Gensim Version ---
    safe_ref = ds_ref.replace('/', '--')
    base_cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'word2vec_torch', cfg.cache_dir))
    cache_root = os.path.join(base_cache_dir, safe_ref)
    
    sw_suffix = 'sw-removed' if cfg.remove_stopwords else 'sw-kept'
    gensim_cache_path = os.path.join(cache_root, f'train.gensim.{sw_suffix}.preprocessed.pkl')

    # --- Data Loading and Preprocessing ---
    corpus, labels = load_and_preprocess_data(data_path, gensim_cache_path, cfg)
    
    # --- Model Training ---
    model = train_word2vec(corpus)
    
    # --- Document Vector Generation ---
    # print("Generating document vectors...")
    # doc_vectors = [get_document_vector(doc_tokens, model) for doc_tokens in tqdm(corpus, desc="Generating document vectors")]
    # X = np.array(doc_vectors)
    # y = labels
    # print("文档向量形状:", X.shape)
    # print("标签形状:", y.shape)
    
    # --- Save and Inspect Model ---
    model_save_path = "word2vec_gensim.model"
    print(f"Saving model to {model_save_path}...")
    model.save(model_save_path)


def test(word: str):
    # --- Load Pre-trained Model ---
    model = Word2Vec.load("word2vec_gensim.model")

    # --- Test with a Sample Word ---
    if word in model.wv:
        try:
            word_embedding = model.wv[word]
            print(f"Embedding vector(shape={word_embedding.shape}) for '{word}': {word_embedding}")
            similar_words = model.wv.most_similar(word)
            print(f"\nThe most similar words to '{word}' are:")
            for w, score in similar_words:
                print(f"{w}: {score:.4f}")
        except KeyError:
            print(f"Word '{word}' not in vocabulary.")
    else:
        print(f"Word '{word}' not in vocabulary.")


if __name__ == "__main__":
    if not os.path.exists("word2vec_gensim.model"):
        train()
    test(input("Input a word: "))
