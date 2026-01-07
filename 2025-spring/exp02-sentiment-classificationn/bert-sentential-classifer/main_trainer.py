import os
import torch
from transformers import (
    BertTokenizerFast, 
    BertForSequenceClassification,
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from config import Config
from dataset import SentimentDataset
from load_data import DataLoader as DataLoaderClass

# 设置 WANDB
os.environ["WANDB_PROJECT"] = "BERT-Sentiment-Trainer"
os.environ["WANDB_LOG_MODEL"] = "false"
os.environ["WANDB_WATCH"] = "false"

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def run_trainer():
    config = Config()

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    print("正在构建数据集...")
    train_dataset = SentimentDataset(
        None, None, None, config.max_seq_length,
        cache_file=f'../cache/train_cache-{config.n_rows}.pt' if config.n_rows else '../cache/train_cache.pt'
    )
    
    val_dataset = SentimentDataset(
        None, None, None, config.max_seq_length,
        cache_file=f'../cache/val_cache-{config.n_rows}.pt' if config.n_rows else '../cache/val_cache.pt'
    )
    
    test_dataset = SentimentDataset(
        None, None, None, config.max_seq_length,
        cache_file='../cache/test_cache.pt'
    )

    # 4. 模型初始化
    print(f"正在加载模型 {config.model_name}...")
    model = BertForSequenceClassification.from_pretrained(
        config.model_name, 
        num_labels=config.num_classes
    )
    
    # 5. 训练参数配置 (基于用户建议的极致加速配置)
    # 注意：Windows下 bitsandbytes (8bit优化器) 通常不可用，这里回退到默认AdamW
    # 如果需要 compile="reduce-overhead"，建议使用静态padding (pad_to_max_length=True)
    # 因为 CUDA Graphs 对动态形状支持不好。但为了通用性，我们先保持动态padding，看看效果。
    
    training_args = TrainingArguments(
        output_dir=config.model_save_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size, # 可以尝试调大，如 32
        per_device_eval_batch_size=config.batch_size * 2,
        learning_rate=config.learning_rate,
        
        # 加速核心配置
        fp16=torch.cuda.is_available(),       # 混合精度训练
        dataloader_num_workers=config.num_workers,
        dataloader_pin_memory=True,
        
        # 优化策略
        gradient_accumulation_steps=1,        # 显存不够时可增加
        warmup_ratio=0.1,
        weight_decay=0.01,
        
        # 日志与评估
        logging_dir='./logs',
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="wandb" if config.LOG_WANDB else "none",
        
        # 编译优化 (PyTorch 2.0)
        # torch_compile=True,   # Trainer自带的compile支持
        # torch_compile_mode="default", # reduce-overhead 在动态shape下可能报错
    )
    
    # 6. 显式编译模型 (用户提到的 reduce-overhead)
    # 注意：如果使用 reduce-overhead，必须确保 shape 尽可能固定。
    # 由于我们使用了 DataCollatorWithPadding (动态padding)，这可能会导致重编译开销。
    # 为了演示，我们先尝试默认模式，如果想极致快，需要配合静态padding。
    if config.use_compile and torch.cuda.is_available():
        print("启用 torch.compile (mode=default)...")
        try:
            model = torch.compile(model, mode="default") # Windows上 default 比较稳
        except Exception as e:
            print(f"Compilation failed: {e}")

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer), # 动态padding
        compute_metrics=compute_metrics
    )
    
    # 8. 开始训练
    print("开始训练...")
    trainer.train()
    
    # 9. 评估测试集
    print("正在评估测试集...")
    test_results = trainer.evaluate(test_dataset)
    print(f"测试集结果: {test_results}")

if __name__ == "__main__":
    run_trainer()
