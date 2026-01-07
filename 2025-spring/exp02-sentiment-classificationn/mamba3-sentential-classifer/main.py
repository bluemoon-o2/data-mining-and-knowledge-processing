import os
import warnings
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer
from config import Config
from dataset import SentimentDataset
from load_data import DataLoader as DataLoaderClass
from model import MambaClassifier
import torch.nn as nn
from tqdm import tqdm
import wandb

warnings.filterwarnings('ignore', category=FutureWarning)

def evaluate(model, eval_loader, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with autocast():
                outputs = model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(outputs, labels)

            _, predictions = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(predictions == labels)
            total_predictions += len(labels)
            total_loss += loss.item()

    return total_loss / len(eval_loader), correct_predictions.double() / total_predictions

def train(train_loader, val_loader, config):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 初始化模型
    model = MambaClassifier(config.model_name, config.num_classes, variant=config.model_variant)
    model.to(device)

    # 编译优化 (Windows 上支持有限)
    if config.use_compile and hasattr(torch, 'compile'):
        print("[Info] Enabling torch.compile...")
        try:
            model = torch.compile(model, backend=config.compile_backend, mode=config.compile_mode)
        except Exception as e:
            print(f"[Warning] torch.compile failed: {e}")

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scaler = GradScaler()

    best_accuracy = 0
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with autocast():
                outputs = model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
            if config.LOG_WANDB:
                wandb.log({"train_loss": loss.item()})

        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader, config)
        
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if config.LOG_WANDB:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss_avg": avg_train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc
            })

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            if not os.path.exists(config.model_save_dir):
                os.makedirs(config.model_save_dir)
            save_path = os.path.join(config.model_save_dir, f"mamba_best_model.pth")
            
            # 如果模型被编译过，保存原始模型
            if hasattr(model, '_orig_mod'):
                model._orig_mod.save_model(save_path)
            else:
                model.save_model(save_path)
            print(f"Saved best model with accuracy: {val_acc:.4f}")

if __name__ == '__main__':
    config = Config()
    
    # 初始化 WandB
    if config.LOG_WANDB:
        try:
            wandb.init(project="sentiment-analysis-mamba", config=config.__dict__)
        except:
            print("[Warning] WandB initialization failed. Logging disabled.")
            config.LOG_WANDB = False

    # 数据加载
    print(f"Loading data from {config.train_path}...")
    data_loader = DataLoaderClass(config)
    
    try:
        train_texts, train_labels = data_loader.load_csv(config.train_path, n=config.n_rows)
        dev_texts, dev_labels = data_loader.load_csv(config.dev_path, n=config.n_rows)
    except FileNotFoundError:
        print(f"Error: Data file not found at {config.train_path}. Please ensure data exists.")
        exit(1)

    # Tokenizer 初始化
    print(f"Loading Tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Mamba/GPT-NeoX tokenizer 通常没有 pad_token，需手动设置
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    # Dataset
    # 使用不同的 cache 文件名以避免冲突
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, config.max_seq_length, cache_file='cache_train_mamba.pt')
    val_dataset = SentimentDataset(dev_texts, dev_labels, tokenizer, config.max_seq_length, cache_file='cache_dev_mamba.pt')

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    train(train_loader, val_loader, config)
