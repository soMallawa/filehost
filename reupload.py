import os
import json
import time
import logging
import torch
import numpy as np
import hashlib
import re
from tqdm import tqdm
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler

# -------- CONFIGURATION --------
DIRECTORY_PATH = "/root/messages"  # Update with your folder path
CHECKPOINT_FILE = "resume_checkpoint.json"  # Stores processed message IDs
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_3eBUzc_GmdvFhHndN4zAXejwE715zbC99jhjLyxjgn9Dxwdxc5Fwq4yBPQCBKKXsUqKzzP")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "febchatv3")
PINECONE_REGION = "us-east-1"  # Change if needed
BATCH_SIZE = 1000  # Max batch size for Pinecone

# -------- LOGGING SETUP --------
log_handler = RotatingFileHandler("upload_log.txt", maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[log_handler, logging.StreamHandler()]
)

# -------- CHECK GPU AVAILABILITY --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Running on device: {DEVICE}")

# -------- INITIALIZE PINECONE --------
logging.info("Initializing Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists, otherwise create it
existing_indexes = pc.list_indexes().names()
if PINECONE_INDEX_NAME not in existing_indexes:
    logging.info(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,  # Adjust based on embedding model
        metric="cosine",
        spec={"serverless": {"cloud": "aws", "region": PINECONE_REGION}}
    )

index = pc.Index(PINECONE_INDEX_NAME)
logging.info("Connected to Pinecone.")

# -------- LOAD EMBEDDING MODEL --------
logging.info("Loading Sentence Transformer model for embeddings...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device=DEVICE)
logging.info("Model loaded successfully.")

# -------- FAILSAFE: LOAD CHECKPOINT DATA --------
def load_checkpoint():
    """Loads resume checkpoint to avoid duplicate processing."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as file:
                return set(json.load(file))
        except Exception as e:
            logging.warning(f"Could not load checkpoint: {e}")
    return set()

uploaded_message_ids = load_checkpoint()

def save_checkpoint():
    """Saves the processed message IDs to resume in case of failure."""
    try:
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as file:
            json.dump(list(uploaded_message_ids), file)
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {e}")

def clean_text(text):
    """Removes non-ASCII characters and trims text."""
    return text.encode("ascii", "ignore").decode().strip()

def generate_safe_id(sender, timestamp):
    """Generates a safe ASCII ID for Pinecone."""
    safe_sender = clean_text(sender)

    # Replace "Kith.mp3" (even if it has emojis) with "Kithmini"
    if "Kith.mp3" in sender:
        safe_sender = "Kithmini"

    if not safe_sender:  # If empty, generate a hash-based ID
        safe_sender = hashlib.md5(sender.encode()).hexdigest()[:10]  # Short hash

    return f"{safe_sender}_{timestamp}"

def process_file(file_path):
    """Extracts messages, generates embeddings, and uploads them to Pinecone in batches."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        messages = data.get("messages", [])
        vectors = []

        logging.info(f"Processing file: {file_path} - Found {len(messages)} messages.")

        for message in tqdm(messages, desc=f"Processing {file_path}", unit="msg"):
            content = clean_text(message.get("content", ""))
            timestamp = message.get("timestamp_ms", 0)
            sender = clean_text(message.get("sender_name", ""))

            # Ensure sender replacement
            if "Kith.mp3" in sender:
                sender = "Kithmini"

            message_id = generate_safe_id(sender, timestamp)

            # Skip already processed messages
            if message_id in uploaded_message_ids:
                continue

            # Ignore empty messages or shared content
            if "share" in message or not content:
                continue

            # Generate text embedding
            embedding = model.encode(content, convert_to_tensor=True).cpu().numpy().tolist()

            # Create vector for Pinecone
            vectors.append({
                "id": message_id,
                "values": embedding,
                "metadata": {
                    "sender": sender,
                    "timestamp": timestamp,
                    "text": content
                }
            })

            # Mark as processed
            uploaded_message_ids.add(message_id)

        # Upload data in batches
        if vectors:
            upload_batches(vectors)
            save_checkpoint()  # Save progress

    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")

def upload_batches(vectors):
    """Uploads data to Pinecone in efficient batches."""
    logging.info(f"Uploading {len(vectors)} messages to Pinecone...")
    
    for i in tqdm(range(0, len(vectors), BATCH_SIZE), desc="Uploading to Pinecone", unit="batch"):
        batch = vectors[i:i + BATCH_SIZE]
        index.upsert(batch)

    logging.info("All batches uploaded successfully.")

def process_all_files():
    """Reads all JSON files in the directory and processes them."""
    files = [f for f in os.listdir(DIRECTORY_PATH) if f.endswith(".json")]

    if not files:
        logging.warning("No JSON files found in the directory.")
        return

    for file in tqdm(files, desc="Processing files", unit="file"):
        process_file(os.path.join(DIRECTORY_PATH, file))
        time.sleep(1)  # Rate-limiting

    logging.info("All messages uploaded successfully!")

# -------- EXECUTE SCRIPT --------
if __name__ == "__main__":
    process_all_files()
