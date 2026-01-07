# Data Configuration
DB_FILE = "./data/medical_knowledge.db"

# Model Configuration
EMBEDDING_MODEL_NAME = 'BAAI/bge-small-zh-v1.5'
EMBEDDING_DIM = 512

# Rerank Configuration
USE_RERANKER = True
RERANK_MODEL_NAME = 'BAAI/bge-reranker-base'

# Indexing and Search Parameters
MAX_ARTICLES_TO_INDEX = 200000
TOP_K = 10
INDEX_METRIC_TYPE = "L2"
INDEX_TYPE = "IVF_FLAT"        # 倒排索引，适合大规模数据
INDEX_PARAMS = {"nlist": 1024} # 增加聚类中心数
SEARCH_PARAMS = {"nprobe": 20} # 搜索时检查的聚类数

USE_HYBRID_RETRIEVAL = True
HYBRID_DENSE_K = 50
HYBRID_SPARSE_K = 100
HYBRID_RRF_K0 = 60

USE_ADAPTIVE_TOPK = True
ADAPTIVE_TOPK_MIN = 10
ADAPTIVE_TOPK_MAX = 50
ADAPTIVE_DISTANCE_GAP_THRESHOLD = 0.15

USE_EVIDENCE_ABSTAIN = True
ABSTAIN_MIN_DOCS = 3
ABSTAIN_MAX_DISTANCE = 1.2

# Generation Parameters (API)
MAX_NEW_TOKENS_GEN = 1536
TEMPERATURE = 0.7
TOP_P = 0.9 
