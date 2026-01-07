class Config:
    """
    模型配置类，包含所有可配置参数
    """
    # 模型参数
    # 使用 HuggingFace 兼容的 Mamba 模型
    # 也可以尝试 larger models: "state-spaces/mamba-370m-hf", "state-spaces/mamba-790m-hf"
    model_name = "state-spaces/mamba-130m-hf"  
    max_seq_length = 128  # 最大序列长度
    num_classes = 2  # 分类类别数量
    
    # 模型变体配置
    # 可选: "default" (last token), "mean_pool", "max_pool", "attention_pool"
    model_variant = "default"

    LOG_WANDB = True
    
    # 训练参数
    batch_size = 32  # Mamba 显存占用情况视实现而定，建议从较小 batch 开始
    learning_rate = 2e-5  # 学习率
    num_epochs = 5  # 训练轮数
    num_workers = 0  # Windows下建议设为0，避免多进程报错
    use_compile = False  # 是否使用 torch.compile
    compile_backend = "inductor"
    compile_mode = "default"
    
    # 路径配置
    n_rows = None
    train_path = "../dataset/train.csv"  # 训练集路径
    dev_path = "../dataset/dev.csv"  # 验证集路径
    test_path = "../dataset/test.csv"  # 测试集路径
    model_save_dir = "../saved_models_mamba"  # 模型保存路径
