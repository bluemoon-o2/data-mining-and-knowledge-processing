import os
import time
import json
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

load_dotenv()

# 配置 HuggingFace 镜像 (加速本地嵌入模型加载)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from config import (
    ABSTAIN_MAX_DISTANCE,
    ABSTAIN_MIN_DOCS,
    EMBEDDING_MODEL_NAME,
    RERANK_MODEL_NAME,
    TOP_K,
    USE_ADAPTIVE_TOPK,
    USE_EVIDENCE_ABSTAIN,
    USE_HYBRID_RETRIEVAL,
    USE_RERANKER,
)
from models import load_embedding_model, load_rerank_model
from faiss_utils import get_faiss_client, index_data_if_needed, retrieve_with_adaptive_topk, search_documents
from rag_core import generate_answer, generate_answer_stream

# 获取当前脚本所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. 初始化 RAG 索引 (本地嵌入模型 + 向量数据库)
print("\n" + "="*50)
print("🔍 正在初始化 RAG 检索环境...")
print("="*50)

faiss_client = get_faiss_client()
embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
rerank_model = None
if USE_RERANKER:
    rerank_model = load_rerank_model(RERANK_MODEL_NAME)

# 直接从数据库和索引初始化
index_data_if_needed(faiss_client, embedding_model)

# 2. 启动 FastAPI
app = FastAPI(title="Medical RAG API (Cloud Version)")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    context: List[dict]
    time_taken: float
    stats: dict = None


async def retrieve_docs(query: str, is_rerank: bool = False) -> List[dict]:
    initial_k = TOP_K * 5 if is_rerank else TOP_K
    try:
        if USE_ADAPTIVE_TOPK and not is_rerank:
            docs, _, _ = retrieve_with_adaptive_topk(faiss_client, query, embedding_model)
            retrieved_docs = docs
        else:
            retrieved_docs, _ = search_documents(
                faiss_client,
                query,
                embedding_model,
                topk=int(initial_k),
                enable_hybrid=bool(USE_HYBRID_RETRIEVAL),
            )
        # 过滤空内容文档，提升后续回答质量
        retrieved_docs = [doc for doc in retrieved_docs if doc.get("content", "").strip()]
        return retrieved_docs
    except Exception as e:
        print(f"文档检索失败: {e}")
        return []

@app.post("/query/stream")
async def query_rag_stream(request: QueryRequest):
    print(f"\n[Stream] 收到问题: {request.query}")
    
    # 1. 搜索 FAISS 并获取文档
    docs = await retrieve_docs(request.query)
    
    if not docs:
        async def empty_gen():
            yield json.dumps({"error": "未找到相关文档。"}) + "\n"
        return StreamingResponse(empty_gen(), media_type="text/event-stream")
    
    # 2. 定义生成器
    async def stream_generator():
        try:
            # 首先发送上下文信息
            yield json.dumps({"context": docs}, ensure_ascii=False) + "\n"
            
            # 使用 run_in_threadpool 防止阻塞事件循环
            from starlette.concurrency import iterate_in_threadpool
            async for chunk_data in iterate_in_threadpool(generate_answer_stream(request.query, docs)):
                if chunk_data:
                    yield json.dumps({
                        "answer_chunk": chunk_data["text"],
                        "token_count": chunk_data["token_count"],
                        "speed": chunk_data["speed"],
                        "elapsed": chunk_data["elapsed"]
                    }, ensure_ascii=False) + "\n"
        except Exception as e:
            print(f"Streaming Error: {e}")
            yield json.dumps({"error": str(e)}, ensure_ascii=False) + "\n"
                
    return StreamingResponse(stream_generator(), media_type="text/event-stream")

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    # 过滤前端心跳检测请求
    if request.query.lower() == "ping":
        return {
            "answer": "pong",
            "context": [],
            "time_taken": 0.0,
            "stats": {"doc_count": faiss_client.doc_count}
        }
    
    print(f"\n[Query] 收到问题: {request.query}")
    start_time = time.time()
    
    # 1. 搜索 FAISS 并获取文档
    docs = await retrieve_docs(request.query, is_rerank=USE_RERANKER)
    
    # 2. 重排序
    if rerank_model and docs:
        print(f"正在重排序 {len(docs)} 条文档...")
        sentence_pairs = [[request.query, doc['content']] for doc in docs]
        scores = rerank_model.predict(sentence_pairs)
        for i, doc in enumerate(docs):
            doc['score'] = float(scores[i])
        docs.sort(key=lambda x: x['score'], reverse=True)
        docs = docs[:TOP_K]
    
    if not docs:
        print("[Query] 未找到相关文档。")
        return QueryResponse(answer="未找到相关文档。", context=[], time_taken=time.time() - start_time)

    if USE_EVIDENCE_ABSTAIN:
        top_dist = None
        try:
            _, dists = search_documents(
                faiss_client,
                request.query,
                embedding_model,
                topk=1,
                enable_hybrid=False,
            )
            if dists:
                top_dist = float(dists[0])
        except Exception:
            top_dist = None
        if len(docs) < int(ABSTAIN_MIN_DOCS) or (top_dist is not None and top_dist > float(ABSTAIN_MAX_DISTANCE)):
            return QueryResponse(
                answer="当前知识库未检索到足够可靠的证据来支持该问题的回答。建议补充关键信息或咨询专业医生。",
                context=docs,
                time_taken=time.time() - start_time,
                stats={"abstained": True, "top_distance": top_dist, "hybrid": bool(USE_HYBRID_RETRIEVAL)},
            )
    
    print(f"[Query] 检索到 {len(docs)} 条相关上下文。")
    
    # 2. 生成答案
    print("[Query] 正在生成答案...")
    result = generate_answer(request.query, docs)
    print("[Query] 答案生成完成。")

    merged_stats = {"hybrid": bool(USE_HYBRID_RETRIEVAL), "adaptive_topk": bool(USE_ADAPTIVE_TOPK), "abstained": False}
    if isinstance(result, dict) and isinstance(result.get("stats"), dict):
        merged_stats.update(result["stats"])
    
    return QueryResponse(
        answer=result["answer"], 
        context=docs, 
        time_taken=time.time() - start_time,
        stats=merged_stats
    )

@app.get("/", response_class=HTMLResponse)
async def get_index():
    print("[Web] 访问首页")
    index_path = os.path.join(CURRENT_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 所有准备就绪！")
    print("👉 请在浏览器访问: http://127.0.0.1:8001")
    print("="*50)
    uvicorn.run(app, host="127.0.0.1", port=8001, workers=1)
