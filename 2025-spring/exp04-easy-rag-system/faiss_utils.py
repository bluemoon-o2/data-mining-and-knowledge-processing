import faiss
import numpy as np
import os
import json
from typing import Dict, List, Tuple

from config import (
    ADAPTIVE_DISTANCE_GAP_THRESHOLD,
    ADAPTIVE_TOPK_MAX,
    ADAPTIVE_TOPK_MIN,
    EMBEDDING_DIM,
    HYBRID_DENSE_K,
    HYBRID_RRF_K0,
    HYBRID_SPARSE_K,
    INDEX_PARAMS,
    INDEX_TYPE,
    SEARCH_PARAMS,
    TOP_K,
    USE_ADAPTIVE_TOPK,
    USE_HYBRID_RETRIEVAL,
)
from db_utils import ensure_fts_index, fts_search, get_docs_by_ids

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
DOC_MAP_PATH = os.path.join(DATA_DIR, "id_to_doc_map.json")
_FTS_READY: bool | None = None

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

class FAISSClient:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index_type = INDEX_TYPE
        self.index = None
        self.doc_count = 0

    def _init_ivf_index(self, num_samples):
        # 动态调整 nlist，确保满足 FAISS 的训练要求 (nx >= nlist)
        # 建议每个聚类中心至少有 30-100 个点，但最小必须满足 1:1
        target_nlist = INDEX_PARAMS.get("nlist", 1024)
        if num_samples < target_nlist:
            print(f"⚠️ 数据量 ({num_samples}) 小于预设聚类数 ({target_nlist})，自动调整 nlist 为 {max(1, num_samples // 10)}")
            nlist = max(1, num_samples // 10)
        else:
            nlist = target_nlist
            
        quantizer = faiss.IndexFlatL2(self.dimension)
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_L2)
        print(f"✅ IVF 索引初始化完成 (nlist={nlist})")

    def add_documents(self, embeddings):
        if len(embeddings) == 0:
            return
        embeddings = np.array(embeddings).astype('float32')
        
        # 如果索引尚未初始化 (针对 IVF)
        if self.index is None:
            if self.index_type == "IVF_FLAT":
                self._init_ivf_index(len(embeddings))
            else:
                self.index = faiss.IndexFlatL2(self.dimension)
        
        if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
            print(f"正在训练 IVF 索引（样本数: {len(embeddings)}）...")
            self.index.train(embeddings)
            
        self.index.add(embeddings)
        self.doc_count += len(embeddings)

    def search(self, query_emb, k=TOP_K):
        query_emb = np.array([query_emb]).astype('float32')
        # 设置搜索参数 (nprobe)
        if isinstance(self.index, faiss.IndexIVFFlat):
            self.index.nprobe = SEARCH_PARAMS.get("nprobe", 10)
            
        distances, indices = self.index.search(query_emb, k)
        return indices[0], distances[0]

    def save(self, index_path, map_path):
        faiss.write_index(self.index, index_path)
        # 不再保存完整的 doc map 到 JSON，只保存数量或轻量元数据
        with open(map_path, 'w', encoding='utf-8') as f:
            json.dump({"doc_count": self.doc_count}, f)

    def load(self, index_path, map_path):
        if os.path.exists(index_path):
            try:
                self.index = faiss.read_index(index_path)
                self.doc_count = self.index.ntotal
                print(f"✅ 从 {index_path} 加载了包含 {self.doc_count} 条向量的索引")
                return True
            except Exception as e:
                print(f"❌ 加载索引失败: {e}")
                return False
        return False

def get_faiss_client():
    """初始化并返回 FAISS 客户端"""
    return FAISSClient(EMBEDDING_DIM)

def index_data_if_needed(client, embedding_model):
    """如果需要，从数据库读取数据并进行索引"""
    # 1. 尝试加载现有索引
    if client.load(FAISS_INDEX_PATH, DOC_MAP_PATH) and client.doc_count > 0:
        return True

    print("🔍 未找到有效索引，正在从数据库读取数据并构建索引...")
    from db_utils import get_doc_count, get_all_docs_minimal
    
    total_docs = get_doc_count()
    if total_docs == 0:
        print("❌ 数据库为空，请先运行 preprocess.py")
        return False

    print(f"正在索引 {total_docs} 条文档（批处理模式）...")
    
    # 分批处理以节省内存
    BATCH_SIZE = 10000
    all_docs = get_all_docs_minimal()
    
    for i in range(0, len(all_docs), BATCH_SIZE):
        batch = all_docs[i:i+BATCH_SIZE]
        texts = [doc['content'] for doc in batch]
        embeddings = embedding_model.encode(texts, show_progress_bar=False)
        client.add_documents(embeddings)
        print(f"  已索引 {min(i + BATCH_SIZE, total_docs)} / {total_docs}...")

    client.save(FAISS_INDEX_PATH, DOC_MAP_PATH)
    print(f"✅ 索引构建完成，共 {client.doc_count} 条。")
    return True

def search_similar_documents(client, query, embedding_model, k=TOP_K):
    """在 FAISS 中搜索相似文档，并从数据库获取详细内容"""
    if not client or client.doc_count == 0:
        return [], []
    
    # BGE 模型建议添加查询指令以提升检索效果
    query = f"为这个句子生成表示以用于检索相关文章：{query}"

    query_emb = embedding_model.encode([query])[0]
    indices, distances = client.search(query_emb, k=k)
    
    # 过滤掉无效索引
    valid_results = [(idx, dist) for idx, dist in zip(indices, distances) if idx != -1]
    if not valid_results:
        print("⚠️ FAISS 未返回任何有效索引 (所有结果均为 -1)")
        return [], []
        
    ids = [int(idx) for idx, dist in valid_results]
    distances = [float(dist) for idx, dist in valid_results]
    
    # 从数据库中按需拉取文档内容
    docs = get_docs_by_ids(ids)
    if not docs:
        print(f"⚠️ 数据库中未找到 ID 列表对应的文档: {ids}")
    else:
        print(f"✅ 成功从数据库获取 {len(docs)} 条文档内容")
    
    return docs, distances


def _rrf_fuse(rank_lists: List[List[int]], *, k0: int, topk: int) -> List[int]:
    scores: Dict[int, float] = {}
    for ranked in rank_lists:
        for r, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / float(k0 + r)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [int(doc_id) for doc_id, _ in fused[:topk]]


def _choose_adaptive_topk(*, query: str, distances: List[float]) -> int:
    if not USE_ADAPTIVE_TOPK:
        return TOP_K
    if not distances:
        return TOP_K
    qlen = len("".join((query or "").split()))
    if qlen <= 12:
        return int(ADAPTIVE_TOPK_MIN)
    if len(distances) >= 2:
        gap = float(distances[1]) - float(distances[0])
        if gap >= float(ADAPTIVE_DISTANCE_GAP_THRESHOLD):
            return int(ADAPTIVE_TOPK_MIN)
    return int(ADAPTIVE_TOPK_MAX)


def search_documents(
    client,
    query: str,
    embedding_model,
    *,
    topk: int = TOP_K,
    enable_hybrid: bool = USE_HYBRID_RETRIEVAL,
) -> Tuple[List[dict], List[float]]:
    if not enable_hybrid:
        return search_similar_documents(client, query, embedding_model, k=topk)

    global _FTS_READY
    if _FTS_READY is None:
        _FTS_READY = bool(ensure_fts_index())
    ok = bool(_FTS_READY)
    dense_k = max(int(topk), int(HYBRID_DENSE_K))
    sparse_k = max(int(topk), int(HYBRID_SPARSE_K))

    dense_docs, dense_dist = search_similar_documents(client, query, embedding_model, k=dense_k)
    dense_ids = []
    dense_id_to_dist: Dict[int, float] = {}
    for d, dist in zip(dense_docs, dense_dist):
        doc_id = int(d.get("id", -1)) if isinstance(d, dict) and "id" in d else None
        if doc_id is None:
            continue
        dense_ids.append(doc_id)
        dense_id_to_dist[doc_id] = float(dist)

    sparse_ids: List[int] = []
    if ok:
        sparse_rows = fts_search(query, sparse_k)
        sparse_ids = [int(doc_id) for doc_id, _ in sparse_rows]

    fused_ids = _rrf_fuse([dense_ids, sparse_ids], k0=int(HYBRID_RRF_K0), topk=int(topk))
    docs = get_docs_by_ids(fused_ids)
    distances = [dense_id_to_dist.get(int(doc_id), float("nan")) for doc_id in fused_ids]
    return docs, distances


def retrieve_with_adaptive_topk(client, query: str, embedding_model) -> Tuple[List[dict], List[float], int]:
    docs, dist = search_documents(client, query, embedding_model, topk=max(int(ADAPTIVE_TOPK_MAX), int(TOP_K)))
    chosen = _choose_adaptive_topk(query=query, distances=[float(x) for x in dist if x == x])
    return docs[:chosen], dist[:chosen], int(chosen)
