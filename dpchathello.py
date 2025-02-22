import os
import json
import time
import logging
import torch
import datetime
import ollama  # Import Ollama package
from tqdm import tqdm
from pinecone import Pinecone  # Corrected Pinecone import
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from logging.handlers import RotatingFileHandler

# -------- CONFIGURATION --------
DIRECTORY_PATH = "/root/messages"
CHECKPOINT_FILE = "resume_checkpoint.json"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_3eBUzc_GmdvFhHndN4zAXejwE715zbC99jhjLyxjgn9Dxwdxc5Fwq4yBPQCBKKXsUqKzzP")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "mmchat_final")
PINECONE_REGION = "us-east-1"
BATCH_SIZE = 50
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/distiluse-base-multilingual-cased-v2"
LLM_MODEL = "deepseek-llm:7b"

# -------- LOGGING SETUP --------
log_handler = RotatingFileHandler("upload_log.txt", maxBytes=10 * 1024 * 1024, backupCount=2, encoding="utf-8")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[log_handler, logging.StreamHandler()]
)

# -------- CHECK GPU AVAILABILITY --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"🚀 Running on GPU device: {DEVICE}")

# -------- INITIALIZE PINECONE --------
logging.info("🔄 Initializing Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

def initialize_pinecone():
    """Initializes Pinecone connection and creates an index if needed."""
    if PINECONE_INDEX_NAME not in pc.list_indexes():
        logging.info(f"✅ Creating Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(PINECONE_INDEX_NAME, spec={"serverless": {"cloud": "aws", "region": PINECONE_REGION}, "dimension": 512, "metric": "cosine"})
    
    return pc.Index(PINECONE_INDEX_NAME)

# -------- LOAD EMBEDDING MODEL --------
def load_embedding_model():
    """Loads the sentence transformer model."""
    logging.info(f"🔄 Loading embedding model: {EMBEDDING_MODEL} on {DEVICE}...")
    model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    logging.info(f"✅ Embedding model {EMBEDDING_MODEL} loaded successfully.")
    return model, tokenizer

# -------- CHECKPOINT HANDLING --------
def load_checkpoint():
    """Loads checkpoint of processed summary IDs to avoid duplicate processing."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as file:
                return set(json.load(file))
        except Exception as e:
            logging.warning(f"⚠️ Could not load checkpoint: {e}")
    return set()

def save_checkpoint(uploaded_message_ids):
    """Saves the processed summary IDs to resume in case of failure."""
    try:
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as file:
            json.dump(list(uploaded_message_ids), file)
        logging.debug("✅ Checkpoint saved successfully.")
    except Exception as e:
        logging.error(f"❌ Failed to save checkpoint: {e}")

# -------- TEXT PROCESSING --------
def clean_text(text):
    """Removes non-ASCII characters and trims text."""
    return text.encode("ascii", "ignore").decode().strip()

def chunk_text(text, tokenizer, max_tokens=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Efficiently splits text into overlapping chunks."""
    tokens = tokenizer.encode(text)
    chunks = [
        tokenizer.decode(tokens[i:i + max_tokens], skip_special_tokens=True)
        for i in range(0, len(tokens), max_tokens - overlap)
    ]
    logging.debug(f"🔹 Chunking completed: {len(chunks)} chunks created.")
    return chunks

def query_ollama(prompt):
    """Generates a response using Ollama's Python API."""
    try:
        start_time = time.time()
        
        # Use Ollama's `generate` function directly
        response = ollama.generate(model=LLM_MODEL, prompt=prompt, options={"max_tokens": 1024, "temperature": 0.7})

        elapsed_time = round(time.time() - start_time, 2)
        
        if response and "response" in response:
            summary_text = response["response"]
            logging.debug(f"✅ LLM response in {elapsed_time} sec.")
            return summary_text
        else:
            logging.error(f"❌ Ollama API returned unexpected response: {response}")
            return ""
    except Exception as e:
        logging.error(f"❌ Error calling Ollama API: {e}")
        return ""

# -------- MAIN PROCESSING FUNCTION --------
def process_all_files(index, model, tokenizer):
    """Reads JSON files, groups messages by day, and processes summaries efficiently."""
    logging.info("🔄 Starting file processing...")
    start_time = time.time()

    uploaded_message_ids = load_checkpoint()
    daily_messages = {}
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
                sender = clean_text(message.get("sender_name", ""))
                content = clean_text(message.get("content", ""))
                daily_messages.setdefault(dt.strftime("%Y-%m-%d"), []).append(f"[{sender}] {content}")
        except Exception as e:
            logging.error(f"❌ Error processing file {file_path}: {e}")

    logging.info(f"🎉 Processing completed in {round(time.time() - start_time, 2)} sec.")
    save_checkpoint(uploaded_message_ids)

# -------- MAIN FUNCTION --------
def main():
    """Main execution function."""
    index = initialize_pinecone()
    model, tokenizer = load_embedding_model()
    process_all_files(index, model, tokenizer)

if __name__ == "__main__":
    main()
