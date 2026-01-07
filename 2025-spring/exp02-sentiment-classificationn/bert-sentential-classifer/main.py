import os
import warnings

import wandb

warnings.filterwarnings('ignore', category=FutureWarning)
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import BertTokenizerFast
from config import Config
from dataset import SentimentDataset
from load_data import DataLoader as DataLoaderClass
from model import SentimentClassifier
import torch.nn as nn
from tqdm import tqdm


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
    # 清理 GPU 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化tokenizer和模型
    model = SentimentClassifier(config.model_name, config.num_classes, variant=config.model_variant)
    model.to(device)

    if config.use_compile:
        print("[Info] 启用 torch.compile 加速...")
        try:
            model = torch.compile(model, backend=config.compile_backend, mode=config.compile_mode)
        except Exception as e:
            print(f"[Warning] torch.compile 失败: {e}")

    num_samples = len(train_dataset)
    total_steps = (num_samples // config.batch_size) * config.num_epochs  # 总迭代步数（向上取整可+1）
    warmup_steps = int(total_steps * 0.05)

    # 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-6,  # 热身起始学习率（极小值，避免初期震荡）
        end_factor=1.0,  # 热身结束后达到初始lr
        total_iters=warmup_steps  # 热身迭代次数
    )

    # 第二步：CosineAnnealingLR实现平滑衰减（热身结束后生效）
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,  # 衰减周期（热身结束后的剩余步数）
        eta_min=1e-7  # 最小学习率（避免衰减到0导致梯度消失）
    )

    # 组合两个调度器（PyTorch 2.0+支持ChainedScheduler，低版本可手动控制）
    scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler_warmup, scheduler_cosine])
    scaler = GradScaler()

    # 训练循环
    best_accuracy = 0
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{config.num_epochs}") as pbar:
            for batch in train_loader:
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
                scheduler.step()
                if config.LOG_WANDB:
                    wandb.log({f"loss": loss.item()})

                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)

        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{config.num_epochs}')
        print(f'Average training loss: {avg_train_loss:.4f}')

        if val_texts is not None and val_labels is not None:
            val_loss, val_accuracy = evaluate(model, val_loader, config)

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                # model.save_model(os.path.join(config.model_save_dir, f"bert-{config.n_rows}.pth"))
            print(f'Validation Accuracy: {val_accuracy:.4f} (Best: {best_accuracy:.4f})')

            if config.LOG_WANDB:
                wandb.log({f"val_loss": val_loss, f"val_accuracy": val_accuracy})

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                print(f"新最佳准确率: {best_accuracy:.4f}")
                # model.save_model(os.path.join(config.model_save_dir, f"bert-{config.n_rows}.pth"))

    return model
if __name__ == "__main__":
    # 设置Hugging Face镜像
    # set_hf_mirrors()

    # 加载配置
    config = Config()

    # 加载数据
    data_loader = DataLoaderClass(config)

    # 分别加载训练集、验证集和测试集
    train_texts, train_labels = data_loader.load_csv("../dataset/train.csv", config.n_rows)
    val_texts, val_labels = data_loader.load_csv("../dataset/dev.csv", config.n_rows)
    test_texts, test_labels = data_loader.load_csv("../dataset/test.csv")

    print(f"训练集样本数: {len(train_texts)}")
    print(f"验证集样本数: {len(val_texts)}")
    print(f"测试集样本数: {len(test_texts)}")

    if config.LOG_WANDB:
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        wandb.init(project="BERT", config=config.__dict__, name=f"{config.n_rows}" if config.n_rows else "default")

    tokenizer = BertTokenizerFast.from_pretrained(config.model_name)

    train_dataset = SentimentDataset(
        train_texts,
        train_labels,
        tokenizer,
        config.max_seq_length,
        cache_file=f'../cache/train_cache-{config.n_rows}.pt' if config.n_rows is not None else f'../cache/train_cache.pt'
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=1
    )

    if val_texts is not None and val_labels is not None:
        val_dataset = SentimentDataset(
            val_texts,
            val_labels,
            tokenizer,
            config.max_seq_length,
            cache_file=f'../cache/val_cache-{config.n_rows}.pt' if config.n_rows is not None else f'../cache/val_cache.pt'
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            prefetch_factor=1
        )

    test_dataset = SentimentDataset(
        test_texts,
        test_labels,
        tokenizer,
        config.max_seq_length,
        cache_file=f'../cache/test_cache.pt'
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=1
    )

    model = train(train_loader, val_loader, config)
    loss, test_accuracy = evaluate(model, test_loader, config)
    print(f"测试集损失: {loss:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")

    if config.LOG_WANDB:
        wandb.finish()
