import os

def load_embedding_model(model_name):
    """加载用于 RAG 检索的嵌入模型"""
    print(f"正在加载嵌入模型: {model_name}...")
    try:
        from sentence_transformers import SentenceTransformer
        
        # 确保使用镜像站
        if not os.environ.get('HF_ENDPOINT'):
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            
        model = SentenceTransformer(model_name)
        print("✅ 嵌入模型加载完成。")
        return model
    except Exception as e:
        print(f"❌ 加载嵌入模型失败: {e}")
        return None

def load_rerank_model(model_name):
    """加载重排序模型"""
    print(f"正在加载重排序模型: {model_name}...")
    try:
        from sentence_transformers import CrossEncoder

        # 确保使用镜像站
        if not os.environ.get('HF_ENDPOINT'):
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            
        model = CrossEncoder(model_name, max_length=512)
        print("✅ 重排序模型加载完成。")
        return model
    except Exception as e:
        print(f"❌ 加载重排序模型失败: {e}")
        return None
