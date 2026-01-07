import sqlite3
from typing import List, Tuple
from config import DB_FILE

def init_db():
    """初始化数据库表结构"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # id 对应 FAISS 中的索引，content 存储实际文本块，metadata 存储标题和来源
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            title TEXT,
            content TEXT,
            source_file TEXT,
            chunk_index INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def get_doc_count():
    """获取数据库中的文档总数"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM documents')
    count = cursor.fetchone()[0]
    conn.close()
    return count

def get_all_docs_minimal():
    """获取所有文档的精简信息（按 ID 排序，确保与 FAISS 索引顺序一致）"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # 显式按 ID 排序，确保 FAISS 索引 ID 与数据库 ID 完美对应
    cursor.execute('SELECT id, content FROM documents ORDER BY id ASC')
    rows = cursor.fetchall()
    conn.close()
    
    return [{"id": row[0], "content": row[1]} for row in rows]

def save_docs_to_db(docs):
    """批量保存文档到数据库"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # 准备插入数据
    data = []
    for i, doc in enumerate(docs):
        # 兼容 abstract 和 content 两个键名
        content = doc.get('content') or doc.get('abstract') or ''
        data.append((
            doc.get('id', i), 
            doc.get('title', ''), 
            content, 
            doc.get('source_file', ''), 
            doc.get('chunk_index', 0)
        ))
    
    # 使用 REPLACE 确保幂等性
    cursor.executemany('''
        INSERT OR REPLACE INTO documents (id, title, content, source_file, chunk_index)
        VALUES (?, ?, ?, ?, ?)
    ''', data)
    
    conn.commit()
    conn.close()

def get_doc_by_id(doc_id):
    """根据 ID 从数据库获取文档（按需查询，省内存）"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('SELECT title, content, source_file FROM documents WHERE id = ?', (int(doc_id),))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "id": int(doc_id),
            "title": row[0],
            "content": row[1],
            "source_file": row[2]
        }
    return None

def get_docs_by_ids(doc_ids):
    """批量根据 IDs 获取文档"""
    if not doc_ids:
        return []
    
    terminal = sqlite3.connect(DB_FILE)
    cursor = terminal.cursor()
    # 占位符 ?,?,? 防止 SQL 注入
    placeholders = ','.join(['?'] * len(doc_ids))
    QUERY = f'SELECT id, title, content, source_file FROM documents WHERE id IN ({placeholders})'
    cursor.execute(QUERY, [int(idx) for idx in doc_ids])
    rows = cursor.fetchall()
    terminal.close()
    
    # 按照请求的顺序排序返回
    id_map = {row[0]: {"id": int(row[0]), "title": row[1], "content": row[2], "source_file": row[3]} for row in rows}
    return [id_map[id] for id in (int(idx) for idx in doc_ids) if id in id_map]


def ensure_fts_index() -> bool:
    conn = sqlite3.connect(DB_FILE)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts
            USING fts5(content, title, source_file, content='documents', content_rowid='id')
            """
        )
        cur.execute("INSERT INTO documents_fts(documents_fts) VALUES('rebuild')")
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def fts_search(query: str, limit: int) -> List[Tuple[int, float]]:
    if not query or limit <= 0:
        return []
    conn = sqlite3.connect(DB_FILE)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT rowid, bm25(documents_fts) AS score
            FROM documents_fts
            WHERE documents_fts MATCH ?
            ORDER BY score ASC
            LIMIT ?
            """,
            (query, int(limit)),
        )
        rows = cur.fetchall()
        return [(int(r[0]), float(r[1])) for r in rows]
    finally:
        conn.close()
