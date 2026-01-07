import torch.nn as nn
from transformers import BertModel
import torch
import torch.nn.functional as F

class LinearAttentionPooling(nn.Module):
    """
    线性注意力池化层 (Linear Attention Pooling)
    利用 Attention 机制将序列特征 (B, L, D) 聚合为全局特征 (B, D)
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, last_hidden_state, attention_mask):
        # last_hidden_state: (B, L, D)
        # attention_mask: (B, L)
        
        # 计算注意力分数: (B, L, 1)
        scores = self.attention(last_hidden_state)
        
        # 处理 mask: 将 pad 位置的分数设为极小值
        mask = attention_mask.unsqueeze(-1) # (B, L, 1)
        scores = scores.masked_fill(mask == 0, -1e9)
        
        # softmax 归一化: (B, L, 1)
        weights = F.softmax(scores, dim=1)
        
        # 加权求和: (B, D)
        context = torch.sum(last_hidden_state * weights, dim=1)
        return context

class GatedAttention(nn.Module):
    """
    门控注意力机制 (Gated Attention)
    使用门控机制来调节特征流
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.gate = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, features):
        # features: (B, D) (通常是 pooled_output)
        gate_values = torch.sigmoid(self.gate(features))
        return features * gate_values

class SwiGLU(nn.Module):
    """
    SwiGLU FFN 变体
    GLU 的一种变体，使用 Swish (SiLU) 激活函数
    """
    def __init__(self, hidden_size, intermediate_size=None):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = hidden_size * 4
            
        self.w1 = nn.Linear(hidden_size, intermediate_size)
        self.w2 = nn.Linear(hidden_size, intermediate_size)
        self.w3 = nn.Linear(intermediate_size, hidden_size)
        
    def forward(self, x):
        # x: (B, D)
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

class SentimentClassifier(nn.Module):
    def __init__(self, model_name, num_classes, variant="default"):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.variant = variant
        self.hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        
        print(f"[Model] Initializing SentimentClassifier with variant: {variant}")
        
        # 变体模块初始化
        if variant == "linear_att":
            self.att_pooling = LinearAttentionPooling(self.hidden_size)
        elif variant == "gated_att":
            self.gated_att = GatedAttention(self.hidden_size)
        elif variant == "ffn_swiglu":
            self.swiglu = SwiGLU(self.hidden_size)
        
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # outputs[0]: last_hidden_state (B, L, D)
        # outputs[1]: pooler_output (B, D) - [CLS] 经过 Tanh
        
        if self.variant == "linear_att":
            # 使用线性注意力聚合序列特征
            last_hidden_state = outputs[0]
            feature = self.att_pooling(last_hidden_state, attention_mask)
            
        elif self.variant == "gated_att":
            # 使用门控机制增强 pooler_output
            pooler_output = outputs[1]
            feature = self.gated_att(pooler_output)
            
        elif self.variant == "ffn_swiglu":
            # 使用 SwiGLU 处理 pooler_output
            pooler_output = outputs[1]
            feature = self.swiglu(pooler_output)
            
        else: # default
            # 标准 BERT 做法
            feature = outputs[1]
            
        output = self.dropout(feature)
        return self.classifier(output)
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))