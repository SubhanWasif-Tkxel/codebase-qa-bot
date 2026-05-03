import os
from dotenv import load_dotenv

load_dotenv()

# Thresholds
RETRIEVAL_CONFIDENCE_THRESHOLD = 0.50
SYNTHESIS_CONFIDENCE_THRESHOLD = 0.70
MAX_RETRIEVAL_ATTEMPTS = 2
TOP_K_CHUNKS = 8

# Chunking
FIXED_CHUNK_SIZE = 400
FIXED_CHUNK_OVERLAP = 80
MAX_CHUNK_TOKENS = 1500

# Embedding
VOYAGE_MODEL = "voyage-code-3"
EMBED_BATCH_SIZE = 64

# GitHub MCP
MCP_SERVER_CMD = ["npx", "-y", "@modelcontextprotocol/server-github"]
RECENT_COMMITS_LIMIT = 20

# Groq
GROQ_MODEL = "llama-3.3-70b-versatile"   # main synthesis model
GROQ_FAST_MODEL = "llama-3.1-8b-instant"  # cheap calls: rephrase, clarify, issue draft
GROQ_MAX_TOKENS = 4096

# Languages with tree-sitter support
TREESITTER_LANGUAGES = {"python", "javascript", "typescript", "go", "rust"}

# File extensions to skip
SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".ttf", ".woff", ".woff2",
    ".pdf", ".zip", ".tar", ".gz", ".lock", ".sum", ".bin", ".exe", ".so",
    ".dylib", ".dll", ".pyc", ".pyo", ".min.js", ".map",
}

# Directories to skip
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", "vendor", "dist", "build",
    ".next", ".nuxt", "coverage", ".pytest_cache", ".mypy_cache", "venv",
    ".venv", "env", ".env",
}

MAX_FILE_BYTES = 500_000

# Paths
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./.chroma")
REPOS_CACHE_DIR = os.getenv("REPOS_CACHE_DIR", "./.repos")

# API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

SYSTEM_PROMPT = """\
You are a codebase expert assistant. Your job is to answer questions about source code \
by citing specific files and line numbers from the provided context.

Rules:
- Always cite your sources using the format [[file_path:line_number]]
- If you reference a specific function or class, cite the line where it is defined
- Be precise and technical — name exact variables, functions, and conditions
- If the provided context is insufficient to answer confidently, say so explicitly
- At the end of every response, output your confidence as: <confidence>0.XX</confidence>
  where 0.0 = complete guess, 1.0 = certain from the code

Never make up file paths or line numbers that are not in the provided context.\
"""
