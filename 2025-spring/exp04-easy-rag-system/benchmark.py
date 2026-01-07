import argparse
import json
import math
import random
import sqlite3
import time
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import faiss
import numpy as np

from config import DB_FILE, EMBEDDING_MODEL_NAME, RERANK_MODEL_NAME
from models import load_embedding_model, load_rerank_model


@dataclass
class EvalResult:
    n: int
    recall_at: Dict[int, float]
    mrr: float
    avg_latency_ms: float
    extras: Dict[str, float] = field(default_factory=dict)


@dataclass
class QAPair:
    doc_id: int
    question: str
    answer: str


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.split())


def _extract_qa(text: str) -> Optional[Tuple[str, str]]:
    if not text:
        return None
    q_pos = text.find("问题：")
    a_pos = text.find("回答：")
    if q_pos == -1 or a_pos == -1 or a_pos <= q_pos:
        return None
    q = _normalize_text(text[q_pos + len("问题：") : a_pos].strip())
    a = _normalize_text(text[a_pos + len("回答：") :].strip())
    if not q or not a:
        return None
    return q, a


def _prepare_query_text(model_name: str, query: str) -> str:
    if "bge" in (model_name or "").lower():
        return "为这个句子生成表示以用于检索相关文章：" + query
    return query


def _sample_ids(total: int, n: int, seed: int) -> List[int]:
    n = min(n, total)
    rng = random.Random(seed)
    return rng.sample(range(total), n)


def _fetch_qa_pairs_by_ids(
    conn: sqlite3.Connection,
    ids: Sequence[int],
    *,
    max_question_chars: int,
    min_question_chars: int,
    max_answer_chars: int,
    min_answer_chars: int,
) -> List[QAPair]:
    if not ids:
        return []
    placeholders = ",".join(["?"] * len(ids))
    query = f"SELECT id, content FROM documents WHERE id IN ({placeholders})"
    cur = conn.execute(query, [int(i) for i in ids])
    rows = cur.fetchall()
    out: List[QAPair] = []
    for doc_id, content in rows:
        qa = _extract_qa(content or "")
        if not qa:
            continue
        q, a = qa
        if len(q) < min_question_chars or len(a) < min_answer_chars:
            continue
        if len(q) > max_question_chars:
            q = q[:max_question_chars]
        if len(a) > max_answer_chars:
            a = a[:max_answer_chars]
        out.append(QAPair(doc_id=int(doc_id), question=q, answer=a))
    return out


def _rank_of(target_id: int, ranked_ids: Sequence[int]) -> Optional[int]:
    for i, rid in enumerate(ranked_ids, start=1):
        if int(rid) == int(target_id):
            return i
    return None


def _compute_metrics(ranks: List[Optional[int]], ks: Sequence[int], latencies_ms: List[float]) -> EvalResult:
    valid_ranks = [r for r in ranks if r is not None]
    recall_at: Dict[int, float] = {}
    for k in ks:
        hit = sum(1 for r in ranks if r is not None and r <= k)
        recall_at[int(k)] = hit / len(ranks) if ranks else 0.0
    mrr = sum(1.0 / r for r in valid_ranks) / len(ranks) if ranks else 0.0
    avg_latency_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    return EvalResult(n=len(ranks), recall_at=recall_at, mrr=mrr, avg_latency_ms=avg_latency_ms, extras={})


def _rrf_fuse(rank_lists: List[List[int]], *, k0: int, topk: int) -> List[int]:
    scores: Dict[int, float] = {}
    for ranked in rank_lists:
        for r, doc_idx in enumerate(ranked, start=1):
            scores[int(doc_idx)] = scores.get(int(doc_idx), 0.0) + 1.0 / float(k0 + r)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [int(doc_idx) for doc_idx, _ in fused[:topk]]


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


def _encode_corpus_answers(embedding_model, model_name: str, answers: Sequence[str], batch_size: int) -> np.ndarray:
    embs = embedding_model.encode(list(answers), batch_size=batch_size, show_progress_bar=False)
    embs = np.asarray(embs, dtype="float32")
    return _l2_normalize(embs)


def _encode_query(embedding_model, model_name: str, question: str, *, use_query_instruction: bool = True) -> np.ndarray:
    q_text = _prepare_query_text(model_name, question) if use_query_instruction else question
    emb = embedding_model.encode([q_text], show_progress_bar=False)
    emb = np.asarray(emb, dtype="float32")
    emb = _l2_normalize(emb)
    return emb[0]


def _build_faiss_ip_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def _tokenize_chars(text: str, max_chars: int) -> List[str]:
    if not text:
        return []
    s = "".join(text.split())
    if max_chars > 0:
        s = s[:max_chars]
    return list(s)


def _bm25_build(corpus: Sequence[str], *, max_doc_chars: int, k1: float, b: float) -> Tuple[Dict[str, List[Tuple[int, int]]], np.ndarray, float, Dict[str, float]]:
    postings: Dict[str, List[Tuple[int, int]]] = {}
    doc_lens = np.zeros(len(corpus), dtype=np.int32)
    df: Dict[str, int] = {}

    for i, doc in enumerate(corpus):
        tokens = _tokenize_chars(doc, max_doc_chars)
        doc_lens[i] = len(tokens)
        if not tokens:
            continue
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        for t, c in tf.items():
            postings.setdefault(t, []).append((i, c))
        for t in tf.keys():
            df[t] = df.get(t, 0) + 1

    n_docs = len(corpus)
    avgdl = float(doc_lens.mean()) if n_docs else 0.0
    idf: Dict[str, float] = {}
    for t, dfi in df.items():
        idf[t] = math.log((n_docs - dfi + 0.5) / (dfi + 0.5) + 1.0)
    return postings, doc_lens, avgdl, idf


def _bm25_search(
    postings: Dict[str, List[Tuple[int, int]]],
    doc_lens: np.ndarray,
    avgdl: float,
    idf: Dict[str, float],
    query: str,
    *,
    max_query_chars: int,
    k1: float,
    b: float,
    topk: int,
) -> List[int]:
    q_tokens = _tokenize_chars(query, max_query_chars)
    if not q_tokens:
        return []
    q_tf: Dict[str, int] = {}
    for t in q_tokens:
        q_tf[t] = q_tf.get(t, 0) + 1

    scores = np.zeros(len(doc_lens), dtype=np.float32)
    for t, qcnt in q_tf.items():
        plist = postings.get(t)
        if not plist:
            continue
        t_idf = idf.get(t, 0.0)
        for doc_idx, tf in plist:
            dl = float(doc_lens[doc_idx])
            denom = tf + k1 * (1.0 - b + b * (dl / (avgdl + 1e-12)))
            scores[doc_idx] += float(t_idf) * (tf * (k1 + 1.0)) / (denom + 1e-12)

    if topk >= len(scores):
        ranked = np.argsort(-scores)
        return [int(i) for i in ranked.tolist()]
    idx = np.argpartition(-scores, topk)[:topk]
    idx = idx[np.argsort(-scores[idx])]
    return [int(i) for i in idx.tolist()]


def evaluate_suite(
    *,
    corpus_size: int,
    num_queries: int,
    seed: int,
    candidate_k: int,
    eval_ks: Sequence[int],
    max_question_chars: int,
    min_question_chars: int,
    max_answer_chars: int,
    min_answer_chars: int,
    embed_batch_size: int,
    rerank_batch_size: int,
    bm25_max_doc_chars: int,
    bm25_max_query_chars: int,
    bm25_k1: float,
    bm25_b: float,
    minilm_model: str,
    bge_model: str,
    rerank_model: str,
) -> Dict[str, Tuple[EvalResult, Optional[EvalResult]]]:
    conn = sqlite3.connect(DB_FILE)
    try:
        cur = conn.execute("SELECT COUNT(*) FROM documents")
        total_docs = int(cur.fetchone()[0])
        if total_docs <= 0:
            raise RuntimeError("数据库为空，请先运行 preprocess.py 构建数据。")

        chosen_ids = _sample_ids(total_docs, max(corpus_size * 3, num_queries * 6), seed)
        qa_pairs = _fetch_qa_pairs_by_ids(
            conn,
            chosen_ids,
            max_question_chars=max_question_chars,
            min_question_chars=min_question_chars,
            max_answer_chars=max_answer_chars,
            min_answer_chars=min_answer_chars,
        )

        rng = random.Random(seed)
        rng.shuffle(qa_pairs)
        qa_pairs = qa_pairs[:corpus_size]
        if len(qa_pairs) < max(100, int(corpus_size * 0.7)):
            raise RuntimeError(f"有效 QA 样本不足：{len(qa_pairs)}（期望 corpus_size={corpus_size}）")

        qa_pairs = qa_pairs[:corpus_size]
        queries = qa_pairs[: min(num_queries, len(qa_pairs))]
        corpus_ids = [p.doc_id for p in qa_pairs]
        corpus_answers = [p.answer for p in qa_pairs]
        id_to_corpus_index = {int(doc_id): int(i) for i, doc_id in enumerate(corpus_ids)}

        results: Dict[str, Tuple[EvalResult, Optional[EvalResult]]] = {}

        max_k = max(int(max(eval_ks)), int(candidate_k), 200)

        bm25_postings, bm25_doc_lens, bm25_avgdl, bm25_idf = _bm25_build(
            corpus_answers, max_doc_chars=bm25_max_doc_chars, k1=bm25_k1, b=bm25_b
        )
        bm25_ranks: List[Optional[int]] = []
        bm25_lat_ms: List[float] = []
        bm25_ranked_lists: List[List[int]] = []
        for p in queries:
            t0 = time.time()
            ranked_idx = _bm25_search(
                bm25_postings,
                bm25_doc_lens,
                bm25_avgdl,
                bm25_idf,
                p.question,
                max_query_chars=bm25_max_query_chars,
                k1=bm25_k1,
                b=bm25_b,
                topk=max_k,
            )
            t1 = time.time()
            bm25_lat_ms.append((t1 - t0) * 1000.0)
            target_idx = id_to_corpus_index.get(int(p.doc_id))
            if target_idx is None:
                bm25_ranks.append(None)
            else:
                bm25_ranks.append(_rank_of(target_idx, ranked_idx[: max(eval_ks)]))
            bm25_ranked_lists.append([int(i) for i in ranked_idx[:max_k]])
        results["BM25-Char"] = (_compute_metrics(bm25_ranks, eval_ks, bm25_lat_ms), None)

        def run_embedding_only(label: str, model_name: str) -> EvalResult:
            embedding_model = load_embedding_model(model_name)
            if embedding_model is None:
                raise RuntimeError(f"嵌入模型加载失败：{model_name}")
            doc_embs = _encode_corpus_answers(embedding_model, model_name, corpus_answers, embed_batch_size)
            index = _build_faiss_ip_index(doc_embs)

            ranks: List[Optional[int]] = []
            lat_ms: List[float] = []
            for p in queries:
                t0 = time.time()
                q_emb = _encode_query(embedding_model, model_name, p.question)
                scores, idx = index.search(np.asarray([q_emb], dtype="float32"), max(eval_ks))
                ranked = [int(i) for i in idx[0].tolist() if int(i) != -1]
                t1 = time.time()
                lat_ms.append((t1 - t0) * 1000.0)
                target_idx = id_to_corpus_index.get(int(p.doc_id))
                ranks.append(_rank_of(target_idx, ranked) if target_idx is not None else None)
            return _compute_metrics(ranks, eval_ks, lat_ms)

        results["MiniLM"] = (run_embedding_only("MiniLM", minilm_model), None)

        embedding_model = load_embedding_model(bge_model)
        if embedding_model is None:
            raise RuntimeError(f"嵌入模型加载失败：{bge_model}")
        doc_embs = _encode_corpus_answers(embedding_model, bge_model, corpus_answers, embed_batch_size)
        index = _build_faiss_ip_index(doc_embs)

        target_indices = [id_to_corpus_index.get(int(p.doc_id)) for p in queries]
        q_embs_inst = np.asarray(
            [_encode_query(embedding_model, bge_model, p.question, use_query_instruction=True) for p in queries], dtype="float32"
        )
        q_embs_noinst = np.asarray(
            [_encode_query(embedding_model, bge_model, p.question, use_query_instruction=False) for p in queries], dtype="float32"
        )

        bge_ranks: List[Optional[int]] = []
        bge_lat_ms: List[float] = []
        bge_ranked_lists: List[List[int]] = []
        bge_noinst_ranks: List[Optional[int]] = []
        bge_noinst_lat_ms: List[float] = []
        bge_noinst_ranked_lists: List[List[int]] = []
        for p in queries:
            t0 = time.time()
            q_emb = _encode_query(embedding_model, bge_model, p.question, use_query_instruction=True)
            _, idx = index.search(np.asarray([q_emb], dtype="float32"), max_k)
            ranked = [int(i) for i in idx[0].tolist() if int(i) != -1]
            t1 = time.time()
            bge_lat_ms.append((t1 - t0) * 1000.0)
            target_idx = id_to_corpus_index.get(int(p.doc_id))
            bge_ranks.append(_rank_of(target_idx, ranked[: max(eval_ks)]) if target_idx is not None else None)
            bge_ranked_lists.append(ranked[:max_k])
        results["BGE"] = (_compute_metrics(bge_ranks, eval_ks, bge_lat_ms), None)

        for p in queries:
            t0 = time.time()
            q_emb = _encode_query(embedding_model, bge_model, p.question, use_query_instruction=False)
            _, idx = index.search(np.asarray([q_emb], dtype="float32"), max_k)
            ranked = [int(i) for i in idx[0].tolist() if int(i) != -1]
            t1 = time.time()
            bge_noinst_lat_ms.append((t1 - t0) * 1000.0)
            target_idx = id_to_corpus_index.get(int(p.doc_id))
            bge_noinst_ranks.append(_rank_of(target_idx, ranked[: max(eval_ks)]) if target_idx is not None else None)
            bge_noinst_ranked_lists.append(ranked[:max_k])
        results["BGE(no-inst)"] = (_compute_metrics(bge_noinst_ranks, eval_ks, bge_noinst_lat_ms), None)

        hybrid_ranks: List[Optional[int]] = []
        hybrid_lat_ms: List[float] = []
        for i, p in enumerate(queries):
            t0 = time.time()
            fused = _rrf_fuse([bm25_ranked_lists[i], bge_ranked_lists[i]], k0=60, topk=max_k)
            t1 = time.time()
            hybrid_lat_ms.append(float(bm25_lat_ms[i]) + float(bge_lat_ms[i]) + (t1 - t0) * 1000.0)
            target_idx = id_to_corpus_index.get(int(p.doc_id))
            hybrid_ranks.append(_rank_of(target_idx, fused[: max(eval_ks)]) if target_idx is not None else None)
        results["Hybrid-RRF"] = (_compute_metrics(hybrid_ranks, eval_ks, hybrid_lat_ms), None)

        hybrid_noinst_ranks: List[Optional[int]] = []
        hybrid_noinst_lat_ms: List[float] = []
        for i, p in enumerate(queries):
            t0 = time.time()
            fused = _rrf_fuse([bm25_ranked_lists[i], bge_noinst_ranked_lists[i]], k0=60, topk=max_k)
            t1 = time.time()
            hybrid_noinst_lat_ms.append(float(bm25_lat_ms[i]) + float(bge_noinst_lat_ms[i]) + (t1 - t0) * 1000.0)
            target_idx = id_to_corpus_index.get(int(p.doc_id))
            hybrid_noinst_ranks.append(_rank_of(target_idx, fused[: max(eval_ks)]) if target_idx is not None else None)
        results["Hybrid-RRF(no-inst)"] = (_compute_metrics(hybrid_noinst_ranks, eval_ks, hybrid_noinst_lat_ms), None)

        rerank_model_obj = load_rerank_model(rerank_model)
        if rerank_model_obj is None:
            raise RuntimeError(f"重排序模型加载失败：{rerank_model}")

        retrieval_ranks: List[Optional[int]] = []
        retrieval_lat_ms: List[float] = []
        rerank_ranks: List[Optional[int]] = []
        rerank_lat_ms: List[float] = []
        bge_reranked_lists: List[List[int]] = []

        for p in queries:
            t0 = time.time()
            q_emb = _encode_query(embedding_model, bge_model, p.question, use_query_instruction=True)
            scores, idx = index.search(np.asarray([q_emb], dtype="float32"), candidate_k)
            cand_idx = [int(i) for i in idx[0].tolist() if int(i) != -1]
            t1 = time.time()
            retrieval_lat_ms.append((t1 - t0) * 1000.0)
            target_idx = id_to_corpus_index.get(int(p.doc_id))
            retrieval_ranks.append(_rank_of(target_idx, cand_idx[: max(eval_ks)]) if target_idx is not None else None)

            t2 = time.time()
            pairs = [[p.question, corpus_answers[i]] for i in cand_idx]
            if not pairs:
                rerank_ranks.append(None)
                rerank_lat_ms.append((time.time() - t2) * 1000.0)
                bge_reranked_lists.append([])
                continue
            r_scores = rerank_model_obj.predict(pairs, batch_size=rerank_batch_size)
            scored = list(zip(cand_idx, [float(s) for s in r_scores]))
            scored.sort(key=lambda x: x[1], reverse=True)
            reranked = [int(i) for i, _ in scored]
            rerank_ranks.append(_rank_of(target_idx, reranked[: max(eval_ks)]) if target_idx is not None else None)
            rerank_lat_ms.append((time.time() - t2) * 1000.0)
            bge_reranked_lists.append(reranked[:max_k])

        retrieval = _compute_metrics(retrieval_ranks, eval_ks, retrieval_lat_ms)
        rerank = _compute_metrics(rerank_ranks, eval_ks, rerank_lat_ms)
        results["BGE+Rerank"] = (retrieval, rerank)

        noinst_retrieval_ranks: List[Optional[int]] = []
        noinst_retrieval_lat_ms: List[float] = []
        noinst_rerank_ranks: List[Optional[int]] = []
        noinst_rerank_lat_ms: List[float] = []
        bge_noinst_reranked_lists: List[List[int]] = []
        for p in queries:
            t0 = time.time()
            q_emb = _encode_query(embedding_model, bge_model, p.question, use_query_instruction=False)
            scores, idx = index.search(np.asarray([q_emb], dtype="float32"), candidate_k)
            cand_idx = [int(i) for i in idx[0].tolist() if int(i) != -1]
            t1 = time.time()
            noinst_retrieval_lat_ms.append((t1 - t0) * 1000.0)
            target_idx = id_to_corpus_index.get(int(p.doc_id))
            noinst_retrieval_ranks.append(_rank_of(target_idx, cand_idx[: max(eval_ks)]) if target_idx is not None else None)

            t2 = time.time()
            pairs = [[p.question, corpus_answers[i]] for i in cand_idx]
            if not pairs:
                noinst_rerank_ranks.append(None)
                noinst_rerank_lat_ms.append((time.time() - t2) * 1000.0)
                bge_noinst_reranked_lists.append([])
                continue
            r_scores = rerank_model_obj.predict(pairs, batch_size=rerank_batch_size)
            scored = list(zip(cand_idx, [float(s) for s in r_scores]))
            scored.sort(key=lambda x: x[1], reverse=True)
            reranked = [int(i) for i, _ in scored]
            noinst_rerank_ranks.append(_rank_of(target_idx, reranked[: max(eval_ks)]) if target_idx is not None else None)
            noinst_rerank_lat_ms.append((time.time() - t2) * 1000.0)
            bge_noinst_reranked_lists.append(reranked[:max_k])

        noinst_retrieval = _compute_metrics(noinst_retrieval_ranks, eval_ks, noinst_retrieval_lat_ms)
        noinst_rerank = _compute_metrics(noinst_rerank_ranks, eval_ks, noinst_rerank_lat_ms)
        results["BGE+Rerank(no-inst)"] = (noinst_retrieval, noinst_rerank)

        hybrid_retrieval_ranks: List[Optional[int]] = []
        hybrid_retrieval_lat_ms: List[float] = []
        hybrid_rerank_ranks: List[Optional[int]] = []
        hybrid_rerank_lat_ms: List[float] = []
        hybrid_reranked_lists: List[List[int]] = []
        for i, p in enumerate(queries):
            fuse_t0 = time.time()
            fused = _rrf_fuse([bm25_ranked_lists[i], bge_ranked_lists[i]], k0=60, topk=max_k)
            fuse_ms = (time.time() - fuse_t0) * 1000.0
            cand_idx = fused[: int(candidate_k)]
            target_idx = id_to_corpus_index.get(int(p.doc_id))
            hybrid_retrieval_ranks.append(_rank_of(target_idx, cand_idx[: max(eval_ks)]) if target_idx is not None else None)
            hybrid_retrieval_lat_ms.append(float(bm25_lat_ms[i]) + float(bge_lat_ms[i]) + float(fuse_ms))

            t1 = time.time()
            pairs = [[p.question, corpus_answers[j]] for j in cand_idx]
            if not pairs:
                hybrid_rerank_ranks.append(None)
                hybrid_rerank_lat_ms.append((time.time() - t1) * 1000.0)
                hybrid_reranked_lists.append([])
                continue
            r_scores = rerank_model_obj.predict(pairs, batch_size=rerank_batch_size)
            scored = list(zip(cand_idx, [float(s) for s in r_scores]))
            scored.sort(key=lambda x: x[1], reverse=True)
            reranked = [int(j) for j, _ in scored]
            hybrid_rerank_ranks.append(_rank_of(target_idx, reranked[: max(eval_ks)]) if target_idx is not None else None)
            hybrid_rerank_lat_ms.append((time.time() - t1) * 1000.0)
            hybrid_reranked_lists.append(reranked[:max_k])

        hybrid_retrieval = _compute_metrics(hybrid_retrieval_ranks, eval_ks, hybrid_retrieval_lat_ms)
        hybrid_rerank = _compute_metrics(hybrid_rerank_ranks, eval_ks, hybrid_rerank_lat_ms)
        results["Hybrid-RRF+Rerank"] = (hybrid_retrieval, hybrid_rerank)

        hybrid_noinst_retrieval_ranks: List[Optional[int]] = []
        hybrid_noinst_retrieval_lat_ms: List[float] = []
        hybrid_noinst_rerank_ranks: List[Optional[int]] = []
        hybrid_noinst_rerank_lat_ms: List[float] = []
        hybrid_noinst_reranked_lists: List[List[int]] = []
        for i, p in enumerate(queries):
            fuse_t0 = time.time()
            fused = _rrf_fuse([bm25_ranked_lists[i], bge_noinst_ranked_lists[i]], k0=60, topk=max_k)
            fuse_ms = (time.time() - fuse_t0) * 1000.0
            cand_idx = fused[: int(candidate_k)]
            target_idx = id_to_corpus_index.get(int(p.doc_id))
            hybrid_noinst_retrieval_ranks.append(_rank_of(target_idx, cand_idx[: max(eval_ks)]) if target_idx is not None else None)
            hybrid_noinst_retrieval_lat_ms.append(float(bm25_lat_ms[i]) + float(bge_noinst_lat_ms[i]) + float(fuse_ms))

            t1 = time.time()
            pairs = [[p.question, corpus_answers[j]] for j in cand_idx]
            if not pairs:
                hybrid_noinst_rerank_ranks.append(None)
                hybrid_noinst_rerank_lat_ms.append((time.time() - t1) * 1000.0)
                hybrid_noinst_reranked_lists.append([])
                continue
            r_scores = rerank_model_obj.predict(pairs, batch_size=rerank_batch_size)
            scored = list(zip(cand_idx, [float(s) for s in r_scores]))
            scored.sort(key=lambda x: x[1], reverse=True)
            reranked = [int(j) for j, _ in scored]
            hybrid_noinst_rerank_ranks.append(_rank_of(target_idx, reranked[: max(eval_ks)]) if target_idx is not None else None)
            hybrid_noinst_rerank_lat_ms.append((time.time() - t1) * 1000.0)
            hybrid_noinst_reranked_lists.append(reranked[:max_k])

        hybrid_noinst_retrieval = _compute_metrics(hybrid_noinst_retrieval_ranks, eval_ks, hybrid_noinst_retrieval_lat_ms)
        hybrid_noinst_rerank = _compute_metrics(hybrid_noinst_rerank_ranks, eval_ks, hybrid_noinst_rerank_lat_ms)
        results["Hybrid-RRF+Rerank(no-inst)"] = (hybrid_noinst_retrieval, hybrid_noinst_rerank)

        def _attach_ctx_metrics(
            result: EvalResult,
            *,
            ranked_lists: List[List[int]],
            query_embs: np.ndarray,
            topk: int,
        ) -> None:
            rel_scores: List[float] = []
            support_scores: List[float] = []
            for qi, ranked in enumerate(ranked_lists):
                tgt = target_indices[qi]
                if tgt is None:
                    continue
                top = ranked[:topk]
                if not top:
                    continue
                qv = query_embs[qi]
                rel_scores.append(float(np.mean([float(np.dot(qv, doc_embs[j])) for j in top])))
                tv = doc_embs[int(tgt)]
                support_scores.append(float(max(float(np.dot(tv, doc_embs[j])) for j in top)))
            result.extras[f"ctx_rel@{topk}"] = float(np.mean(rel_scores)) if rel_scores else 0.0
            result.extras[f"ctx_support@{topk}"] = float(np.mean(support_scores)) if support_scores else 0.0

        topk_ctx = int(max(eval_ks)) if eval_ks else 10
        _attach_ctx_metrics(results["BM25-Char"][0], ranked_lists=bm25_ranked_lists, query_embs=q_embs_inst, topk=topk_ctx)
        _attach_ctx_metrics(results["BGE"][0], ranked_lists=bge_ranked_lists, query_embs=q_embs_inst, topk=topk_ctx)
        _attach_ctx_metrics(results["BGE(no-inst)"][0], ranked_lists=bge_noinst_ranked_lists, query_embs=q_embs_noinst, topk=topk_ctx)

        hybrid_lists_inst = [_rrf_fuse([bm25_ranked_lists[i], bge_ranked_lists[i]], k0=60, topk=max_k) for i in range(len(queries))]
        hybrid_lists_noinst = [
            _rrf_fuse([bm25_ranked_lists[i], bge_noinst_ranked_lists[i]], k0=60, topk=max_k) for i in range(len(queries))
        ]
        _attach_ctx_metrics(results["Hybrid-RRF"][0], ranked_lists=hybrid_lists_inst, query_embs=q_embs_inst, topk=topk_ctx)
        _attach_ctx_metrics(results["Hybrid-RRF(no-inst)"][0], ranked_lists=hybrid_lists_noinst, query_embs=q_embs_noinst, topk=topk_ctx)

        _attach_ctx_metrics(results["BGE+Rerank"][1], ranked_lists=bge_reranked_lists, query_embs=q_embs_inst, topk=topk_ctx)
        _attach_ctx_metrics(results["BGE+Rerank(no-inst)"][1], ranked_lists=bge_noinst_reranked_lists, query_embs=q_embs_noinst, topk=topk_ctx)
        _attach_ctx_metrics(results["Hybrid-RRF+Rerank"][1], ranked_lists=hybrid_reranked_lists, query_embs=q_embs_inst, topk=topk_ctx)
        _attach_ctx_metrics(results["Hybrid-RRF+Rerank(no-inst)"][1], ranked_lists=hybrid_noinst_reranked_lists, query_embs=q_embs_noinst, topk=topk_ctx)

        return results
    finally:
        conn.close()


def _format_suite_table(results: Dict[str, Tuple[EvalResult, Optional[EvalResult]]], eval_ks: Sequence[int]) -> str:
    ks = [int(k) for k in eval_ks]
    headers = ["方法"] + [f"Recall@{k}" for k in ks] + ["MRR", "平均延迟(ms)"]
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join([":---"] + [":---:" for _ in headers[1:]]) + " |")

    order = ["BM25-Char", "MiniLM", "BGE", "Hybrid-RRF", "BGE+Rerank", "Hybrid-RRF+Rerank"]
    for name in order:
        if name not in results:
            continue
        retrieval, rerank = results[name]
        if rerank is None:
            total_lat = retrieval.avg_latency_ms
            row = [name] + [f"{retrieval.recall_at[k]:.3f}" for k in ks] + [f"{retrieval.mrr:.3f}", f"{total_lat:.1f}"]
            out.append("| " + " | ".join(row) + " |")
        else:
            total_lat = retrieval.avg_latency_ms + rerank.avg_latency_ms
            row = [name] + [f"{rerank.recall_at[k]:.3f}" for k in ks] + [f"{rerank.mrr:.3f}", f"{total_lat:.1f}"]
            out.append("| " + " | ".join(row) + " |")

    return "\n".join(out)


def main() -> None:
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-size", type=int, default=5000)
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--eval-ks", type=str, default="5,10")
    parser.add_argument("--max-question-chars", type=int, default=256)
    parser.add_argument("--min-question-chars", type=int, default=8)
    parser.add_argument("--max-answer-chars", type=int, default=1024)
    parser.add_argument("--min-answer-chars", type=int, default=16)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--rerank-batch-size", type=int, default=16)
    parser.add_argument("--bm25-max-doc-chars", type=int, default=512)
    parser.add_argument("--bm25-max-query-chars", type=int, default=256)
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--minilm-model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--bge-model", type=str, default=EMBEDDING_MODEL_NAME)
    parser.add_argument("--rerank-model", type=str, default=RERANK_MODEL_NAME)
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    eval_ks = [int(x.strip()) for x in args.eval_ks.split(",") if x.strip()]
    if not eval_ks:
        raise RuntimeError("eval-ks 不能为空")

    results = evaluate_suite(
        corpus_size=args.corpus_size,
        num_queries=args.num_queries,
        seed=args.seed,
        candidate_k=args.candidate_k,
        eval_ks=eval_ks,
        max_question_chars=args.max_question_chars,
        min_question_chars=args.min_question_chars,
        max_answer_chars=args.max_answer_chars,
        min_answer_chars=args.min_answer_chars,
        embed_batch_size=args.embed_batch_size,
        rerank_batch_size=args.rerank_batch_size,
        bm25_max_doc_chars=args.bm25_max_doc_chars,
        bm25_max_query_chars=args.bm25_max_query_chars,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        minilm_model=args.minilm_model,
        bge_model=args.bge_model,
        rerank_model=args.rerank_model,
    )

    print(
        " ".join(
            [
                f"corpus_size={args.corpus_size}",
                f"num_queries={args.num_queries}",
                f"seed={args.seed}",
                f"candidate_k={args.candidate_k}",
                f"minilm={args.minilm_model}",
                f"bge={args.bge_model}",
                f"rerank={args.rerank_model}",
            ]
        )
    )
    print(_format_suite_table(results, eval_ks))

    if args.output_json:
        def _pack_result(ret: EvalResult) -> Dict[str, object]:
            return {
                "n": int(ret.n),
                "recall_at": {str(k): float(v) for k, v in ret.recall_at.items()},
                "mrr": float(ret.mrr),
                "avg_latency_ms": float(ret.avg_latency_ms),
                "extras": {str(k): float(v) for k, v in (ret.extras or {}).items()},
            }

        def _use_final(name: str) -> Optional[EvalResult]:
            if name not in results:
                return None
            ret, rr = results[name]
            return rr if rr is not None else ret

        k_main = int(eval_ks[0]) if eval_ks else 5
        def _ab_row(res: Optional[EvalResult], *, total_lat_ms: float) -> Dict[str, float]:
            if res is None:
                return {"Recall@5": 0.0, "MRR": 0.0, "Latency(ms)": float(total_lat_ms)}
            return {"Recall@5": float(res.recall_at.get(k_main, 0.0)), "MRR": float(res.mrr), "Latency(ms)": float(total_lat_ms)}

        def _total_lat(name: str) -> float:
            if name not in results:
                return 0.0
            ret, rr = results[name]
            return float(ret.avg_latency_ms) + (float(rr.avg_latency_ms) if rr is not None else 0.0)

        ablation = {
            "Full": _ab_row(_use_final("Hybrid-RRF+Rerank"), total_lat_ms=_total_lat("Hybrid-RRF+Rerank")),
            "-Hybrid": _ab_row(_use_final("BGE+Rerank"), total_lat_ms=_total_lat("BGE+Rerank")),
            "-Rerank": _ab_row(_use_final("BGE"), total_lat_ms=_total_lat("BGE")),
            "-QueryInstruction": _ab_row(_use_final("Hybrid-RRF+Rerank(no-inst)"), total_lat_ms=_total_lat("Hybrid-RRF+Rerank(no-inst)")),
        }

        payload = {
            "args": {
                "corpus_size": int(args.corpus_size),
                "num_queries": int(args.num_queries),
                "seed": int(args.seed),
                "candidate_k": int(args.candidate_k),
                "eval_ks": [int(k) for k in eval_ks],
                "minilm_model": str(args.minilm_model),
                "bge_model": str(args.bge_model),
                "rerank_model": str(args.rerank_model),
            },
            "results": {
                name: {
                    "retrieval": _pack_result(ret),
                    "rerank": None
                    if rr is None
                    else _pack_result(rr),
                }
                for name, (ret, rr) in results.items()
            },
            "ablation": ablation,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
