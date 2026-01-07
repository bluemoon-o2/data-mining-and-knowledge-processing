import os
import json
from db_utils import init_db, save_docs_to_db, get_doc_count, get_all_docs_minimal
from datasets import load_dataset

# 设置 HuggingFace 镜像以加速下载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
output_json_path = os.path.join(DATA_DIR, 'processed_data.json')
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def split_text(text, chunk_size=512, chunk_overlap=50):
    """
    将文本分割成指定大小的块，并带有重叠。
    """
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
        if start >= len(text):
             break
    
    return [c.strip() for c in chunks if c.strip()]



def run_preprocessing(huatuo_limit=200000, force_refresh=False):
    """
    直接从 HuggingFace 数据集加载并处理，存入数据库。
    支持数据库缓存：如果 DB 中已有数据，则跳过处理。
    """
    import sys
    if "--force_refresh" in sys.argv:
        force_refresh = True
        
    print(f"🚀 开始预处理检查...")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 初始化数据库
    init_db()
    
    # 检查数据库中是否已有数据 (如果少于 1000 条，可能数据不完整，强制重新处理)
    current_count = get_doc_count()
    if current_count > 1000 and not force_refresh:
        print(f"📦 数据库已存在 {current_count} 条数据，跳过数据集下载。")
        
        # 检查 JSON 索引是否存在，如果不存在则从 DB 导出
        if not os.path.exists(output_json_path):
            print("正在从数据库导出精简 JSON 索引...")
            all_data_minimal = get_all_docs_minimal()
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(all_data_minimal, f, ensure_ascii=False)
            print(f"✅ JSON 索引导出完成。")
        return

    all_data = []
    chunk_count = 0

    print(f"正在从 HuggingFace 加载 Huatuo26M-Lite 数据集...")
    try:
        # 使用默认缓存路径
        ds = load_dataset(
            "FreedomIntelligence/Huatuo26M-Lite", 
            split='train', 
            streaming=False
        )
        
        huatuo_count = 0
        total_to_process = min(len(ds), huatuo_limit)
        print(f"找到数据集记录 {len(ds)} 条，准备处理前 {total_to_process} 条...")

        for i in range(total_to_process):
            entry = ds[i]
            question = entry.get('question', '')
            answer = entry.get('answer', '')
            
            if not question or not answer:
                continue
            
            title = f"医疗问答：{question}" # 移除 20 字符限制，存储全长问题
            content = f"问题：{question}\n回答：{answer}"
            
            # 文本分割
            chunks = split_text(content, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            
            for idx, chunk in enumerate(chunks):
                chunk_count += 1
                data_entry = {
                    "id": chunk_count - 1,
                    "title": title,
                    "content": chunk,
                    "source_file": "Huatuo26M-Lite (Cached)",
                    "chunk_index": idx
                }
                all_data.append(data_entry)
            
            huatuo_count += 1
            if huatuo_count % 10000 == 0:
                print(f"  已处理 {huatuo_count} / {total_to_process} 条记录...")
        
        print(f"✅ 数据集记录处理完成，共计 {huatuo_count} 条，生成 {chunk_count} 个文本块。")
        
    except Exception as e:
        print(f"❌ 加载或处理数据集时出错: {e}")
        return

    # 2. 保存到数据库 (SQLite)
    print("正在保存文本内容到数据库...")
    save_docs_to_db(all_data)
    
    # 3. 保存精简版索引 (JSON) 供 FAISS 启动时快速构建 ID 映射
    print(f"正在保存精简索引到: {output_json_path}")
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            # 只保留 id 和用于检索的 content 字段，极大减小 JSON 体积
            minimal_data = [{"id": d["id"], "content": d["content"]} for d in all_data]
            json.dump(minimal_data, f, ensure_ascii=False)
        print(f"✅ 预处理完成！总块数: {chunk_count}")
    except Exception as e:
        print(f"错误：无法写入 JSON 文件: {e}")

if __name__ == "__main__":
    run_preprocessing(huatuo_limit=200000)
