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
from logging.handlers import RotatingFileHandler

# -------- CONFIGURATION --------
DIRECTORY_PATH = "/root/messages"
CHECKPOINT_FILE = "resume_checkpoint.json"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_3eBUzc_GmdvFhHndN4zAXejwE715zbC99jhjLyxjgn9Dxwdxc5Fwq4yBPQCBKKXsUqKzzP")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "mmmchat")
PINECONE_REGION = "us-east-1"
BATCH_SIZE = 50  # Increase batch size to leverage GPU acceleration on RunPod.io

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
        dimension=384,  
        metric="cosine",
        spec={"serverless": {"cloud": "aws", "region": PINECONE_REGION}}
    )
index = pc.Index(PINECONE_INDEX_NAME)
logging.info("✅ Connected to Pinecone.")

# -------- LOAD EMBEDDING MODEL --------
logging.info("🔄 Loading Sentence Transformer model on GPU...")
model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2", device=DEVICE)
logging.info("✅ Model loaded successfully.")

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

def query_ollama(prompt):
    """
    Sends a prompt to the Ollama LLM and returns the summary.
    """
    url = "http://localhost:11434/api"  # Ensure this endpoint works on RunPod.io
    payload = {
        "prompt": prompt,
        "model": "deepseek-llm:7b",
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

        embedding = model.encode(summary, convert_to_tensor=True).to(DEVICE).cpu().numpy().tolist()  # Keep on GPU
        vector = {"id": summary_id, "values": embedding, "metadata": {"date": day, "senders": senders_list, "summary": summary}}
        batch_vectors.append(vector)

        if len(batch_vectors) >= BATCH_SIZE:
            index.upsert(batch_vectors)
            for v in batch_vectors:
                uploaded_message_ids.add(v["id"])
            save_checkpoint()
            batch_vectors = []  

    if batch_vectors:
        index.upsert(batch_vectors)
        for v in batch_vectors:
            uploaded_message_ids.add(v["id"])
        save_checkpoint()

    total_time = round(time.time() - start_time, 2)
    logging.info(f"🎉 All processing completed in {total_time} seconds on RunPod.io.")

# -------- EXECUTE SCRIPT --------
if __name__ == "__main__":
    process_all_files()
