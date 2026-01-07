import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os

class SentimentDataset(Dataset):
    """
    情感分类数据集类
    
    参数:
        texts (List[str]): 文本列表
        labels (List[int]): 标签列表
        tokenizer: 分词器 (AutoTokenizer)
        max_len (int): 最大序列长度
        cache_file (str, optional): 缓存文件路径
    """
    def __init__(self, texts, labels, tokenizer, max_len, cache_file=None):
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.input_ids = []
        self.attention_masks = []
        
        # 尝试加载缓存
        if cache_file and os.path.exists(cache_file):
            print(f"加载缓存文件: {cache_file}")
            cached_data = torch.load(cache_file)
            self.input_ids = cached_data['input_ids']
            self.attention_masks = cached_data['attention_mask']
            # 验证缓存数据大小是否匹配
            if len(self.input_ids) != len(texts):
                print("警告: 缓存数据大小与当前数据不匹配，重新预处理...")
                self.input_ids = []
                self.attention_masks = []
            else:
                return

        # 批量预处理以加速
        batch_size = 10000
        num_samples = len(texts)
        print(f"正在预处理 {num_samples} 个样本 (CPU Tokenization)...")
        
        # 预分配内存
        self.input_ids = torch.empty((num_samples, max_len), dtype=torch.long)
        self.attention_masks = torch.empty((num_samples, max_len), dtype=torch.long)
        
        for i in tqdm(range(0, num_samples, batch_size), desc="Tokenizing (CPU)"):
            batch_texts = [str(t) for t in texts[i:i + batch_size]]
            # Mamba tokenizer (GPT-NeoX) 通常没有 pad_token，需在外部设置
            encodings = tokenizer.batch_encode_plus(
                batch_texts,
                add_special_tokens=True,
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            # 直接填入预分配的 Tensor
            end_idx = min(i + batch_size, num_samples)
            self.input_ids[i:end_idx] = encodings['input_ids']
            self.attention_masks[i:end_idx] = encodings['attention_mask']
            
        # 保存缓存
        if cache_file:
            print(f"保存缓存到: {cache_file}")
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            torch.save({
                'input_ids': self.input_ids,
                'attention_mask': self.attention_masks
            }, cache_file)
            
    def __len__(self):
        """返回数据集大小"""
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        """
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }
