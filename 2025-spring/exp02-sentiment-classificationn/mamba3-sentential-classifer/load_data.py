import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Union
import json
import os

class DataLoader:
    """
    数据加载器类，用于加载和预处理数据
    """
    def __init__(self, config):
        """
        初始化数据加载器
        
        参数:
            config: 配置对象，包含数据路径等参数
        """
        self.config = config
        
    def load_csv(self, file_path: str, n = None) -> Tuple[List[str], List[int]]:
        """
        加载CSV格式的数据文件
        
        参数:
            file_path (str): CSV文件路径
            
        返回:
            Tuple[List[str], List[int]]: 文本列表和标签列表
        """
        try:
            # 读取CSV文件，不使用列名
            if n is None:
                df = pd.read_csv(file_path, header=None, names=['label', 'title', 'text'])
            else:
                df = pd.read_csv(file_path, header=None, names=['label', 'title', 'text'], nrows=n)
            
            # 将标签从 1,2 转换为 0,1
            df['label'] = df['label'].map({1: 0, 2: 1})
            
            # 合并标题和文本
            texts = [f"{title} {text}" for title, text in zip(df['title'], df['text'])]
            labels = df['label'].tolist()
            
            return texts, labels
        except Exception as e:
            print(f"加载CSV文件时出错: {str(e)}")
            raise
    
    def load_json(self, file_path: str) -> Tuple[List[str], List[int]]:
        """
        加载JSON格式的数据文件
        预期JSON格式为：[{"text": "文本", "label": 0}, ...]
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            texts = [item['text'] for item in data]
            labels = [item['label'] for item in data]
            
            return texts, labels
        except Exception as e:
            print(f"加载JSON文件时出错: {str(e)}")
            raise
    
    def load_txt(self, text_file: str, label_file: str) -> Tuple[List[str], List[int]]:
        """
        加载文本文件
        text_file: 每行一个文本
        label_file: 每行一个标签
        """
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f]
            
            with open(label_file, 'r', encoding='utf-8') as f:
                labels = [int(line.strip()) for line in f]
            
            if len(texts) != len(labels):
                raise ValueError("文本数量与标签数量不匹配")
            
            return texts, labels
        except Exception as e:
            print(f"加载文本文件时出错: {str(e)}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        文本预处理
        
        参数:
            text (str): 输入文本
            
        返回:
            str: 预处理后的文本
        """
        # 去除多余的空白字符
        text = ' '.join(text.split())
        # 可以根据需要添加更多预处理步骤
        return text
