# ── Data Loading ───────────────────────────────────────────────────
FILE_SIZE_WARNING_MB = 50        # Warn user if file exceeds this
FILE_SIZE_LIMIT_MB = 200         # Hard limit — refuse to process above this
CHUNK_SIZE_ROWS = 10_000         # Rows per chunk for large file profiling
LARGE_FILE_THRESHOLD_MB = 20     # Use chunked profiling above this size

# ── Profiling ──────────────────────────────────────────────────────
SAMPLE_SIZE = 100                # Rows sampled for dtype inference
NUMERIC_INFERENCE_THRESHOLD = 0.8    # 80%+ convertible → should_be_numeric
DATETIME_INFERENCE_THRESHOLD = 0.8   # 80%+ convertible → should_be_datetime
CATEGORICAL_CARDINALITY_RATIO = 0.05 # unique/total < 5% → categorical
HIGH_CARDINALITY_THRESHOLD = 20      # unique count above this → high cardinality
TOP_VALUES_COUNT = 5                 # Top N values shown in profile

# ── Skewness ───────────────────────────────────────────────────────
SKEW_SYMMETRIC_MAX = 0.5         # Below this → symmetric
SKEW_MODERATE_MAX = 1.0          # Between symmetric and this → moderate_skew
                                 # Above this → high_skew

# ── Null Handling ──────────────────────────────────────────────────
NULL_DROP_COLUMN_THRESHOLD = 0.60    # Drop column if null% exceeds this

# ── Outlier Handling ───────────────────────────────────────────────
OUTLIER_IQR_MULTIPLIER = 1.5         # Standard IQR fence multiplier
OUTLIER_ZSCORE_THRESHOLD = 3.0       # Z-score threshold for outlier detection
OUTLIER_DROP_MAX_PCT = 0.02          # Only drop rows if outliers < 2% of data
OUTLIER_DROP_MIN_ROWS = 100          # Only drop rows if dataset has 100+ rows

# ── Visualization ──────────────────────────────────────────────────
MIN_ROWS_FOR_KDE = 10            # Minimum rows needed to render KDE
CATEGORICAL_TOP_N = 10           # Top N bars in categorical bar charts
HEATMAP_MIN_COLUMNS = 2          # Minimum numeric columns for heatmap

# ── LLM ───────────────────────────────────────────────────────────
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE_PLAN = 0.1       # Low = deterministic cleaning decisions
LLM_TEMPERATURE_CHAT = 0.3       # Slightly higher for natural conversation
LLM_MAX_TOKENS_PLAN = 2048
LLM_MAX_TOKENS_CHAT = 1024

# ── RAG (V2) ───────────────────────────────────────────────────────
RAG_CHUNK_SIZE = 500             # Characters per context chunk
RAG_CHUNK_OVERLAP = 50           # Overlap between chunks
RAG_TOP_K = 3                    # Top K chunks retrieved per query
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # Free local embedding model