import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TEMPERATURE = 0.2

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_BACKEND = "huggingface"
EMBEDDING_DIM = 1024
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")

QDRANT_DB_PATH = "./knowledge_base/qdrant_store"
QDRANT_COLLECTION_NAME = "class8_science"
RECREATE_COLLECTION_ON_INGEST = os.getenv("RECREATE_COLLECTION_ON_INGEST", "false").lower() == "true"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

CHUNK_LAYER_RECURSIVE = {
    "macro": {"chunk_size": 3600, "chunk_overlap": 650},
    "meso": {"chunk_size": 1900, "chunk_overlap": 320},
    "micro": {"chunk_size": 1100, "chunk_overlap": 180},
}

EMBED_CHUNK_TIERS = ["macro", "meso", "micro"]
ENABLE_TOKEN_MICRO_LAYER = False

TOKEN_MICRO_CHUNK = {
    "encoding": "cl100k_base",
    "chunk_size": 640,
    "chunk_overlap": 96,
}

BENGALI_RECURSIVE_SEPARATORS = [
    "\n\n\n",
    "\n\n",
    "\n",
    "। ",
    "।",
    ". ",
    "! ",
    "? ",
    "; ",
    " — ",
    " – ",
    ", ",
    " ",
    "",
]

PDF_INGESTION_MODE = os.getenv("PDF_INGESTION_MODE", "auto")
PDF_OCR_MIN_TEXT_CHARS = 40
PDF_RENDER_SCALE = 3.0

PDF_VISION_MODEL = "gpt-4o-mini"
PDF_VISION_DETAIL = "high"
PDF_VISION_MAX_TOKENS = 8000
PDF_VISION_TEMPERATURE = 0.0
PDF_PREPROCESS_ENABLE = True
PDF_PREPROCESS_GRAYSCALE = True
PDF_PREPROCESS_AUTOCONTRAST = True
PDF_PREPROCESS_SHARPEN_FACTOR = 1.6
PDF_VISION_SYSTEM_PROMPT = os.getenv("PDF_VISION_SYSTEM_PROMPT") or ""
PDF_VISION_USER_PROMPT = os.getenv("PDF_VISION_USER_PROMPT") or ""

ASSESSMENT_GENERATION_MODE = os.getenv("ASSESSMENT_GENERATION_MODE", "specialized_agents")

ALL_ASSESSMENT_TASK_KEYS = (
    "mcqs",
    "creative_questions",
)

ASSESSMENT_COUNTS = {
    "mcqs": 5,
    "creative_questions": 1,
}

TEACHER_TASK_SUBROOT = os.getenv("TEACHER_TASK_SUBROOT", "tasks")

DIFFICULTY_LEVELS = ("beginner", "intermediate", "advanced")
LEVEL_LABELS_BN = {
    "beginner": "শিক্ষানবিস — সহজ ভাষা, সরাসরি তথ্য ও স্পষ্ট ধাপ",
    "intermediate": "মধ্যবর্তী — মানসম্মত শ্রেণিকক্ষ ভাষা, হালকা যুক্তি",
    "advanced": "উন্নত — গভীর বিশ্লেষণ, তুলনা ও প্রয়োগমূলক চিন্তা চাও",
}

TOP_K_RESULTS = 10
RETRIEVAL_VECTOR_K = 10
RETRIEVAL_BM25_K = 10
RETRIEVAL_FINAL_K = 8
RETRIEVAL_ENABLE_TOPIC_FOCUS = True
RETRIEVAL_TOPIC_HARD_FILTER = False
RETRIEVAL_TOPIC_BOOST = 3.0
RETRIEVAL_MIN_TOPIC_OVERLAP = 1
RETRIEVAL_MIN_QUERY_COVERAGE = 0.18

RETRIEVAL_EXPAND_K = 30
RETRIEVAL_NEIGHBOR_GRAIN_WINDOW = 2
RETRIEVAL_NEIGHBOR_SEQUENCE_WINDOW = 2
RETRIEVAL_GRAPH_HOPS = 2
RETRIEVAL_GRAPH_MAX_NODES = 80
RETRIEVAL_GRAPH_SEED_TOP_N = 4

ABLATE_CCI = False
ABLATE_CCR = False
ABLATE_ROUTING = False
ABLATE_COVERAGE_GATE = False
ABLATE_GRAPH_EXPANSION = False
RETRIEVAL_DENSE_ONLY = False
RETRIEVAL_BM25_ONLY = False

RETRIEVAL_BACKEND = "cope"
VERIFIER = "saver"

BASELINE_COLLECTIONS = {
    "raptor": "class8_science_raptor",
    "graphrag": "class8_science_graphrag",
}

ASSESSMENT_MCQS_MAX_ATTEMPTS = 5
ASSESSMENT_CQ_MAX_ATTEMPTS = 5

PREGEN_ENABLE_COVERAGE_CHECK = True
PREGEN_MIN_CONTEXT_CHUNKS = 2
PREGEN_MIN_DISTINCT_GRAINS = 1
PREGEN_MIN_DISTINCT_HEADINGS = 1
PREGEN_MIN_QUERY_TERM_HITS = 2
PREGEN_MEDIUM_MIN_CONTEXT_CHUNKS = 2
PREGEN_MEDIUM_MIN_DISTINCT_GRAINS = 1
PREGEN_MEDIUM_MIN_QUERY_TERM_HITS = 1
PREGEN_NEIGHBOR_RETRY_ENABLE = True

TEXTBOOK_DIR = "./data/textbooks"
LOGS_DIR = "./logs"
RUN_OUTPUT_DIR = os.getenv("RUN_OUTPUT_DIR", "./output")
TEACHER_SESSION_SUBDIR = os.getenv("TEACHER_SESSION_SUBDIR", "teacher_sessions")

MAX_VERIFICATION_RETRIES = 2
MIN_FAITHFULNESS_SCORE = 0.80
MIN_ANSWER_RELEVANCE_SCORE = 0.75
MAX_HALLUCINATION_RISK_SCORE = 0.35

LLM_PRICING = {
    "gpt-4o-mini": {
        "input":  0.15,
        "output": 0.60,
    }
}
