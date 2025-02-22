import os
import json
import time
import logging
import torch
import datetime
import requests
from tqdm import tqdm
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from logging.handlers import RotatingFileHandler

# -------- CONFIGURATION --------
DIRECTORY_PATH = "/root/messages"
CHECKPOINT_FILE = "resume_checkpoint.json"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_3eBUzc_GmdvFhHndN4zAXejwE715zbC99jhjLyxjgn9Dxwdxc5Fwq4yBPQCBKKXsUqKzzP")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "febchatvf")
PINECONE_REGION = "us-east-1"
BATCH_SIZE = 50  # Optimized batch size for RunPod.io
CHUNK_SIZE = 512  # Max tokens per chunk for embedding
CHUNK_OVERLAP = 50  # Token overlap to preserve context
EMBEDDING_MODEL = "sentence-transformers/distiluse-base-multilingual-cased-v2"  # Best for Singlish embeddings
LLM_MODEL = "mixtral"  # Best LLM for Singlish chat summarization

# Estimated time tracking (updated in runtime)
ESTIMATED_LLM_TIME_PER_DAY = 3  # Initial estimate (will update dynamically)
ESTIMATED_PINECONE_UPSERT_TIME_PER_BATCH = 5  # Initial estimate (updates dynamically)
llm_times = []  # Track real execution times for LLM
pinecone_times = []  # Track real execution times for Pinecone upserts

# -------- LOGGING SETUP --------
log_handler = RotatingFileHandler("upload_log.txt", maxBytes=10 * 1024 * 1024, backupCount=2, encoding="utf-8")
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[log_handler, logging.StreamHandler()]
)

# -------- CHECK GPU AVAILABILITY --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"🚀 Running on GPU device: {DEVICE}")

# -------- INITIALIZE PINECONE --------
logging.info("🔄 Initializing Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
existing_indexes = pc.list_indexes().names()
if PINECONE_INDEX_NAME not in existing_indexes:
    logging.info(f"✅ Creating Pinecone index: {PINECONE_INDEX_NAME}")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=512,  # `distiluse-base-multilingual-cased-v2` generates 512-dim embeddings
        metric="cosine",
        spec={"serverless": {"cloud": "aws", "region": PINECONE_REGION}}
    )
index = pc.Index(PINECONE_INDEX_NAME)
logging.info("✅ Connected to Pinecone.")

# -------- LOAD EMBEDDING MODEL --------
logging.info(f"🔄 Loading embedding model: {EMBEDDING_MODEL} on GPU...")
model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
logging.info(f"✅ Embedding model {EMBEDDING_MODEL} loaded successfully.")

# -------- FAILSAFE: LOAD & SAVE CHECKPOINT --------
def load_checkpoint():
    """Loads checkpoint of processed summary IDs to avoid duplicate processing."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as file:
                return set(json.load(file))
        except Exception as e:
            logging.warning(f"⚠️ Could not load checkpoint: {e}")
    return set()

uploaded_message_ids = load_checkpoint()

def save_checkpoint():
    """Saves the processed summary IDs to resume in case of failure."""
    try:
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as file:
            json.dump(list(uploaded_message_ids), file)
        logging.debug("✅ Checkpoint saved successfully.")
    except Exception as e:
        logging.error(f"❌ Failed to save checkpoint: {e}")

# -------- HELPER FUNCTIONS --------
def clean_text(text):
    """Removes non-ASCII characters and trims text."""
    return text.encode("ascii", "ignore").decode().strip()

def generate_summary_id(day_str):
    """Generates a unique ID for a daily summary."""
    return f"summary_{day_str}"

def chunk_text(text, max_tokens=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Splits text into chunks of `max_tokens` tokens with an `overlap`."""
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens - overlap):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)
    logging.debug(f"🔹 Chunking completed: {len(chunks)} chunks created.")
    return chunks

def query_ollama(prompt):
    """
    Sends a prompt to the Ollama LLM and returns the summary.
    """
    global ESTIMATED_LLM_TIME_PER_DAY

    url = "http://localhost:11434/api"
    payload = {
        "prompt": prompt,
        "model": LLM_MODEL,
        "max_tokens": 1024,
        "temperature": 0.7
    }
    try:
        start_time = time.time()
        response = requests.post(url, json=payload)
        elapsed_time = round(time.time() - start_time, 2)
        llm_times.append(elapsed_time)
        ESTIMATED_LLM_TIME_PER_DAY = sum(llm_times) / len(llm_times)  # Update estimate

        if response.status_code == 200:
            result = response.json()
            summary_text = result.get("completion", "")
            logging.debug(f"✅ LLM response in {elapsed_time} sec. Updated estimate: {ESTIMATED_LLM_TIME_PER_DAY:.2f} sec/day.")
            return summary_text
        else:
            logging.error(f"❌ LLM API returned status {response.status_code}: {response.text}")
            return ""
    except Exception as e:
        logging.error(f"❌ Error calling LLM API: {e}")
        return ""

# -------- MAIN PROCESSING FUNCTION --------
def process_all_files():
    """
    Reads all JSON files, groups messages by day, and processes summaries efficiently.
    """
    logging.info("🔄 Starting processing of all files on RunPod.io...")
    start_time = time.time()

    daily_messages = {}  
    files = [f for f in os.listdir(DIRECTORY_PATH) if f.endswith(".json")]
    
    if not files:
        logging.warning("⚠️ No JSON files found in the directory.")
        return

    estimated_total_time = (len(files) * ESTIMATED_LLM_TIME_PER_DAY) + ((len(files) // BATCH_SIZE) * ESTIMATED_PINECONE_UPSERT_TIME_PER_BATCH)
    logging.info(f"🔹 Estimated processing time: ~{estimated_total_time:.2f} seconds ({estimated_total_time / 60:.2f} min)")

    for file in tqdm(files, desc="📂 Processing files", unit="file"):
        file_path = os.path.join(DIRECTORY_PATH, file)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            messages = data.get("messages", [])
            for message in messages:
                if "share" in message or not message.get("content"):
                    continue
                timestamp = message.get("timestamp_ms", 0)
                dt = datetime.datetime.fromtimestamp(timestamp / 1000)
                sender = clean_text(message.get("sender_name", ""))
                content = clean_text(message.get("content", ""))
                
                msg_info = f"[{sender}] {content}"
                daily_messages.setdefault(dt.strftime("%Y-%m-%d"), []).append(msg_info)
        except Exception as e:
            logging.error(f"❌ Error processing file {file_path}: {e}")

    logging.info(f"🎉 Processing completed in {round(time.time() - start_time, 2)} sec.")
