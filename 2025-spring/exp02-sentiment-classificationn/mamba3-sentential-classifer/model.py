import torch
import torch.nn as nn
from transformers import AutoModel

class MambaClassifier(nn.Module):
    def __init__(self, model_name, num_classes, variant="default"):
        super().__init__()
        # 加载 HF 格式的 Mamba
        # 注意：必须安装 transformers>=4.39.0
        print(f"[Model] Loading Mamba backbone: {model_name}")
        try:
            self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
             print(f"[Error] Failed to load Mamba model. Ensure transformers>=4.39.0 is installed. Error: {e}")
             raise e
             
        self.config = self.backbone.config
        self.hidden_size = self.config.hidden_size
        self.variant = variant
        self.num_classes = num_classes
        
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
        print(f"[Model] Initializing MambaClassifier with variant: {variant}")
        
    def forward(self, input_ids, attention_mask=None):
        # Mamba 模型通常只接受 input_ids
        # 部分实现可能接受 attention_mask，但主要是为了兼容性，内部是 causal 的
        outputs = self.backbone(input_ids=input_ids)
        
        # transformers 的 MambaModel 输出通常包含 last_hidden_state
        last_hidden_state = outputs.last_hidden_state # (B, L, D)
        
        if self.variant == "mean_pool":
            # masked mean pooling
            if attention_mask is None:
                 feature = torch.mean(last_hidden_state, dim=1)
            else:
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
                sum_mask = input_mask_expanded.sum(1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                feature = sum_embeddings / sum_mask
            
        elif self.variant == "max_pool":
            if attention_mask is None:
                feature = torch.max(last_hidden_state, 1)[0]
            else:
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                last_hidden_state[input_mask_expanded == 0] = -1e9 
                feature = torch.max(last_hidden_state, 1)[0]
            
        else: # default: last token pooling
            if attention_mask is not None:
                # 找到最后一个 1 的索引 (Right padding)
                # 注意：如果 attention_mask 是 [1, 1, 0, 0]，sum=2，index=1 (第二个token)
                last_token_indices = attention_mask.sum(dim=1) - 1
                # 防止 index 为 -1
                last_token_indices = torch.clamp(last_token_indices, min=0)
                
                batch_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
                feature = last_hidden_state[batch_indices, last_token_indices]
            else:
                # 假设没有 padding，取最后一个
                 feature = last_hidden_state[:, -1, :]

        output = self.dropout(feature)
        logits = self.classifier(output)
        return logits
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
