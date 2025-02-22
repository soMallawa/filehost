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
BATCH_SIZE = 100 if torch.cuda.is_available() else 50  
VECTOR_DIMENSION = 512  # ✅ Set to match the embedding model

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

if PINECONE_INDEX_NAME in existing_indexes:
    index_info = pc.describe_index(PINECONE_INDEX_NAME)
    if index_info['dimension'] != VECTOR_DIMENSION:
        logging.warning(f"⚠️ Deleting incorrect Pinecone index: {PINECONE_INDEX_NAME}")
        pc.delete_index(PINECONE_INDEX_NAME)

if PINECONE_INDEX_NAME not in existing_indexes:
    logging.info(f"✅ Creating Pinecone index: {PINECONE_INDEX_NAME} with {VECTOR_DIMENSION} dimensions")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=VECTOR_DIMENSION,
        metric="cosine",
        spec={"serverless": {"cloud": "aws", "region": PINECONE_REGION}}  # ✅ FIXED SYNTAX ERROR
    )

index = pc.Index(PINECONE_INDEX_NAME)
logging.info("✅ Pinecone index initialized successfully.")

# -------- LOAD EMBEDDING MODEL --------
logging.info("🔄 Loading Sentence Transformer model on GPU...")
model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2", device=DEVICE)  
logging.info("✅ Model loaded successfully.")

# -------- FAILSAFE: LOAD & SAVE CHECKPOINT --------
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as file:
                return set(json.load(file))
        except Exception as e:
            logging.warning(f"⚠️ Could not load checkpoint: {e}")
    return set()

uploaded_message_ids = load_checkpoint()

def save_checkpoint():
    try:
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as file:
            json.dump(list(uploaded_message_ids), file)
    except Exception as e:
        logging.error(f"❌ Failed to save checkpoint: {e}")

# -------- HELPER FUNCTIONS --------
def clean_text(text):
    return text.encode("ascii", "ignore").decode().strip()

def format_time_for_llm(timestamp):
    dt = datetime.datetime.strptime(timestamp, "%H:%M:%S")
    return dt.strftime("%I:%M %p")  # Converts to "5:30 PM"

def generate_summary_id(day_str):
    return f"summary_{day_str}"

def query_ollama(day, compiled_text):
    """
    Sends a refined prompt to the Ollama LLM to generate a summary with quoted dialogues and key events.
    """
    url = "http://localhost:11434/api/generate"

    prompt = f"""
    Summarize the following conversation logs for {day} in a **concise and engaging** manner. 
    Capture key discussions, emotions, and **include notable direct quotes** where relevant. 
    Use **exact times in 12-hour AM/PM format** (e.g., "5:30 PM") for better context.

    **Log for {day}:**
    {compiled_text}

    **Guidelines:**
    - Summarize the most **important moments** of the day.
    - Maintain a **storytelling flow** but **keep it brief and natural**.
    - **Include at least 2-3 direct quotes** from the conversation.
    - Highlight key debates, jokes, personal reflections, or emotional moments.

    Now, generate a **concise summary** for {day} based on the provided conversation logs, ensuring you include **notable quotes** where appropriate.
    """

    payload = {
        "model": "deepseek-llm:7b",
        "prompt": prompt,
        "max_tokens": 1024,
        "temperature": 0.7,
        "stream": False
    }

    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=15)

        if response.status_code != 200:
            logging.error(f"❌ LLM API returned status {response.status_code}: {response.text}")
            return ""

        try:
            result = response.json()
        except json.JSONDecodeError as e:
            logging.error(f"❌ Invalid JSON from LLM API: {e}\nResponse: {response.text}")
            return ""

        elapsed_time = round(time.time() - start_time, 2)
        logging.info(f"✅ LLM response received in {elapsed_time} sec.")

        return result.get("response", "").strip() if "response" in result else ""

    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Error calling LLM API: {e}")
        return ""

# -------- MAIN PROCESSING FUNCTION --------
def process_all_files():
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
                readable_time = format_time_for_llm(time_str)

                sender = clean_text(message.get("sender_name", ""))
                content = clean_text(message.get("content", ""))

                msg_info = f"[{readable_time}] {sender}: {content}"
                daily_messages.setdefault(day_str, []).append(msg_info)
                daily_senders.setdefault(day_str, set()).add(sender)
        except Exception as e:
            logging.error(f"❌ Error processing file {file_path}: {e}")

    for day, msgs in daily_messages.items():
        summary_id = generate_summary_id(day)
        if summary_id in uploaded_message_ids:
            continue

        summary = query_ollama(day, "\n".join(msgs))
        if not summary:
            continue

        embedding = model.encode(summary, convert_to_tensor=True).cpu().numpy().tolist()
        index.upsert([{"id": summary_id, "values": embedding, "metadata": {"date": day, "summary": summary}}])

    logging.info(f"🎉 All processing completed in {round(time.time() - start_time, 2)} seconds.")

# -------- EXECUTE SCRIPT --------
if __name__ == "__main__":
    process_all_files()
