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
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your_api_key_here")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "febchatv3")
PINECONE_REGION = "us-east-1"
BATCH_SIZE = 50  # Optimized batch size for RunPod.io
CHUNK_SIZE = 512  # Max tokens per chunk for embedding
CHUNK_OVERLAP = 50  # Token overlap to preserve context
EMBEDDING_MODEL = "thenlper/gte-large"  # Best for long-text embeddings

# -------- LOGGING SETUP --------
log_handler = RotatingFileHandler("upload_log.txt", maxBytes=10 * 1024 * 1024, backupCount=2, encoding="utf-8")
logging.basicConfig(
    level=logging.INFO,
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
        dimension=1024,  # `gte-large` generates 1024-dim embeddings
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
    return chunks

def query_ollama(prompt):
    """
    Sends a prompt to the Ollama LLM and returns the summary.
    """
    url = "http://localhost:11434/api"  # Ensure this endpoint works on RunPod.io
    payload = {
        "prompt": prompt,
        "model": "mistral",
        "max_tokens": 1024,
        "temperature": 0.7
    }
    try:
        start_time = time.time()
        response = requests.post(url, json=payload)
        elapsed_time = round(time.time() - start_time, 2)

        if response.status_code == 200:
            result = response.json()
            logging.info(f"✅ LLM response received in {elapsed_time} sec.")
            return result.get("completion", "")
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
    daily_senders = {}   
    files = [f for f in os.listdir(DIRECTORY_PATH) if f.endswith(".json")]
    
    if not files:
        logging.warning("⚠️ No JSON files found in the directory.")
        return

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
                day_str = dt.strftime("%Y-%m-%d")
                time_str = dt.strftime("%H:%M:%S")
                sender = clean_text(message.get("sender_name", ""))
                content = clean_text(message.get("content", ""))
                
                msg_info = f"[{time_str}] {sender}: {content}"
                daily_messages.setdefault(day_str, []).append(msg_info)
                daily_senders.setdefault(day_str, set()).add(sender)
        except Exception as e:
            logging.error(f"❌ Error processing file {file_path}: {e}")

    batch_vectors = []

    for day, msgs in tqdm(daily_messages.items(), desc="📅 Processing daily messages", unit="day"):
        summary_id = generate_summary_id(day)
        if summary_id in uploaded_message_ids:
            logging.info(f"⏭️ Summary for {day} already processed, skipping.")
            continue

        senders_list = sorted(list(daily_senders.get(day, [])))
        compiled_text = f"Date: {day}\nSenders: {', '.join(senders_list)}\nMessages:\n" + "\n".join(msgs)
        
        logging.info(f"📝 Generating summary for {day}...")
        summary = query_ollama(f"Summarize the following messages:\n\n{compiled_text}")
        if not summary:
            logging.error(f"❌ LLM summarization failed for {day}")
            continue

        chunks = chunk_text(summary)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{summary_id}_part_{i}"
            embedding = model.encode(chunk, convert_to_tensor=True).cpu().numpy().tolist()
            vector = {"id": chunk_id, "values": embedding, "metadata": {"date": day, "senders": senders_list, "summary_chunk": chunk}}
            batch_vectors.append(vector)

        if len(batch_vectors) >= BATCH_SIZE:
            index.upsert(batch_vectors)
            batch_vectors = []

    if batch_vectors:
        index.upsert(batch_vectors)

    logging.info(f"🎉 Processing completed in {round(time.time() - start_time, 2)} sec.")
