class Config:
    """
    模型配置类，包含所有可配置参数
    """
    # 模型参数
    model_name = "bert-base-chinese"  # 预训练模型名称
    max_seq_length = 128  # 最大序列长度，超过此长度的文本将被截断
    num_classes = 2  # 分类类别数量，二分类为2
    
    # 模型变体配置
    # 可选: "default", "linear_att", "gated_att", "ffn_swiglu"
    model_variant = "default"

    LOG_WANDB = True
    
    # 训练参数
    batch_size = 256  # 批次大小，可根据GPU内存调整
    learning_rate = 1e-5  # 学习率
    num_epochs = 5  # 训练轮数
    num_workers = 4  # 数据加载器的工作线程数
    use_compile = False  # 是否使用 torch.compile 加速 (PyTorch 2.0+, Windows支持有限)
    compile_backend = "inductor"  # 编译后端: "inductor" (默认), "cudagraphs", "tensorrt" (需额外安装)
    compile_mode = "default"  # 编译模式: "default", "reduce-overhead", "max-autotune" (编译极慢但运行最快)
    
    # 路径配置
    n_rows = None
    train_path = "../dataset/train.csv"  # 训练集路径
    dev_path = "../dataset/dev.csv"  # 验证集路径
    test_path = "../dataset/test.csv"  # 测试集路径
    model_save_dir = "../saved_models"  # 模型保存路径
