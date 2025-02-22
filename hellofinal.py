import os
import json
import time
import logging
import torch
import datetime
import requests
import re
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
BATCH_SIZE = 100 if torch.cuda.is_available() else 50  # Dynamically adjust batch size

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
    """Removes non-ASCII characters, emojis, trims text, and replaces 'kith.mp3' with 'Kithmini'."""
    if text:
        text = text.encode("ascii", "ignore").decode().strip()  # Remove non-ASCII characters
        text = re.sub(r"kith\.mp3", "Kithmini", text, flags=re.IGNORECASE)  # Replace variations of 'kith.mp3'
        text = re.sub(r"[^\w\s]", "", text)  # Remove emojis and special characters
    return text

def generate_summary_id(day_str):
    """Generates a unique ID for a daily summary."""
    return f"summary_{day_str}"

def query_ollama(messages, date, senders):
    """
    Sends a prompt to the Ollama LLM to generate a **detailed, romantic & time-aware** summary of the conversation.
    """
    url = "http://localhost:11434/api/generate"  # Correct API endpoint

    prompt = f"""
    You are an AI that summarizes chat conversations in a **casual, slightly romantic storytelling style**, while keeping all key details intact. 
    Your task is to create a **warm, engaging summary** of the following messages.

    **Conversation Details:**
    - **Date**: {date}
    - **Participants**: {', '.join(senders)}
    - **Messages** (with timestamps):

    {messages}

    **Summary Instructions:**
    - Capture the **main topics** discussed.
    - Include the **times of key moments** (e.g., "At 10:15 AM, Alex teasingly asked...").
    - Preserve **dates, times, sender names, and emotions**.
    - Highlight any **sweet, playful, or intimate moments** in a natural way.
    - Indicate the **tone of the conversation** (e.g., lighthearted, affectionate, deep).
    - Mention any **inside jokes, meaningful exchanges, or thoughtful gestures**.
    - Keep it **genuine, not overly dramatic or forced**—just like reading an old text thread with a smile.

    **Example Format:**
    "On {date}, {', '.join(senders)} exchanged messages filled with warmth.  
    At 09:30 AM, [Sender1] playfully teased [Sender2] about [funny topic], making them laugh.  
    At 10:15 AM, [Sender2] shared something heartfelt about [topic], making the conversation take a more intimate turn.  
    By 11:45 AM, they were reminiscing about [memory] and ended the chat with [a playful goodbye/a warm exchange/a promise to meet soon]."

    Now, generate a **detailed, heartfelt summary** based on these messages.
    """

    payload = {
        "model": "deepseek-llm:7b",
        "prompt": prompt,
        "max_tokens": 2048,
        "temperature": 0.7
    }

    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=15)

        if response.status_code == 200:
            result = response.json()
            elapsed_time = round(time.time() - start_time, 2)
            logging.info(f"✅ LLM response received in {elapsed_time} sec.")
            return result.get("response", "").strip()  # Ensure clean output

        logging.error(f"❌ LLM API returned status {response.status_code}: {response.text}")
        return ""

    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Error calling LLM API: {e}")
        return ""

# -------- MAIN PROCESSING FUNCTION --------
def process_all_files():
    """
    Reads all JSON files, groups messages by day, and processes summaries efficiently.
    """
    logging.info("🔄 Starting processing of all files...")
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
                time_str = dt.strftime("%I:%M %p")
                sender = clean_text(message.get("sender_name", ""))
                content = clean_text(message.get("content", ""))

                msg_info = f"At {time_str}, {sender} said: \"{content}\""
                daily_messages.setdefault(day_str, []).append(msg_info)
                daily_senders.setdefault(day_str, set()).add(sender)
        except Exception as e:
            logging.error(f"❌ Error processing file {file_path}: {e}")

    logging.info("🎉 Processing completed.")

# -------- EXECUTE SCRIPT --------
if __name__ == "__main__":
    process_all_files()
