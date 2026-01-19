import os
import uuid
import pandas as pd
from typing import Set

from embedding import EmbeddingModel, SparseEmbeddingModel
from vector_db import VectorDatabase
import config  # Import configuration

# ============================================================
# 1. Đọc từng file JSONL
# ============================================================
def load_jsonl(file_path: str) -> pd.DataFrame:
    print(f"📄 Đang đọc: {file_path}")
    try:
        df = pd.read_json(file_path, lines=True)
    except Exception as e:
        print(f"❌ Lỗi đọc JSONL: {e}")
        raise e

    required_cols = ["id", "title", "text"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"❌ File {file_path} thiếu cột bắt buộc '{col}'")

    print(f"   → {len(df)} dòng\n")
    return df


# ============================================================
# 2. Build document text để embed
# ============================================================
def build_combine_row(row):
    return (
        f"Tiêu đề: {row['title']}\n"
        f"Nội dung: {row['text']}\n"
    )


# ============================================================
# 3. Lấy danh sách ID đã tồn tại (Deduplication)
# ============================================================
def get_existing_ids(vector_db: VectorDatabase, collection_name: str) -> Set[str]:
    existing_ids = set()
    print(f"🔍 Checking existing docs in {collection_name} ({vector_db.db_type})...")

    try:
        if vector_db.db_type == "qdrant":
            # Check if collection exists first
            if not vector_db.client.collection_exists(collection_name):
                 return set()

            # Scroll to get all IDs
            # Note: For very large collections (millions), getting ALL IDs might be heavy.
            # But for 100-200k, it's reasonable for a simple script.
            next_page_offset = None
            while True:
                results, next_page_offset = vector_db.client.scroll(
                    collection_name=collection_name,
                    limit=2000,
                    with_payload=False,
                    with_vectors=False,
                    offset=next_page_offset
                )
                for r in results:
                    existing_ids.add(str(r.id))
                
                if next_page_offset is None:
                    break

        elif vector_db.db_type == "chromadb":
            collection = vector_db.client.get_or_create_collection(name=collection_name)
            # define logic for chroma if needed, usually .get() with include=['metadatas'] is heavy
            # Chroma logic omitted for brevity as user focuses on Qdrant
            pass
        
        elif vector_db.db_type == "mongodb":
            db = vector_db.client.get_database("vector_db")
            collection = db[collection_name]
            docs = collection.find({}, {"_id": 1}) # Assuming _id is the unique identifier
            existing_ids = {str(d["_id"]) for d in docs}

        elif vector_db.db_type == "supabase":
            resp = vector_db.client.table(collection_name).select("id").execute()
            existing_ids = {str(r["id"]) for r in resp.data}

    except Exception as e:
        print(f"⚠️ Warning during existing check: {e}")

    return existing_ids


# ============================================================
# 4. Insert batch
# ============================================================
def insert_batch(vector_db: VectorDatabase, batch_data: list, collection_name: str):
    try:
        vector_db.insert(data=batch_data, collection_name=collection_name)
        return True, None
    except Exception as e:
        return False, str(e)


import concurrent.futures

# ============================================================
# 5. Xử lý 1 file → 1 collection (Parallel Batch Processing)
# ============================================================
def process_batch_worker(batch_records, embedding_model, sparse_model, vector_db, collection_name, batch_num, file_path):
    """
    Worker function to process a single batch:
    1. Encode texts
    2. Format payloads
    3. Insert to Vector DB
    """
    try:
        # 1. Prepare texts
        texts = [r["combine_text"] for r in batch_records]
        
        # 2. Batch Encode
        embeddings = embedding_model.encode(texts, mode="passage")
        
        # 2.1 Sparse Encode
        sparse_embeddings = []
        if sparse_model:
            sparse_embeddings = sparse_model.encode(texts)
        else:
            # fill None
            sparse_embeddings = [None] * len(texts)
        
        # 3. Construct Data Payload
        batch_data = []
        for j, record in enumerate(batch_records):
            batch_data.append({
                "id": record["uuid"],
                "embedding": embeddings[j],
                "metadata": {
                    "title": record.get("title", ""),
                    "text": record.get("text", ""),
                    "doc_id": str(record["id"]),
                    "index": record.get("original_index", -1),
                    "source": os.path.basename(file_path)
                }
            })
            
            # Add sparse if available
            if sparse_embeddings[j]:
                batch_data[-1]["sparse_embedding"] = sparse_embeddings[j]
            
        # 4. Insert Batch
        ok, err = insert_batch(vector_db, batch_data, collection_name)
        if ok:
            return True, len(batch_data), batch_num
        else:
            return False, f"Batch {batch_num}: {err}", batch_num

    except Exception as e:
        return False, f"Batch {batch_num} Exception: {str(e)}", batch_num


def process_single_file(vector_db, embedding, sparse_model, file_path, collection_name, batch_size=256, max_workers=4, reset_collection=False):
    print("\n" + "="*80)
    print(f"🚀 PROCESSING COLLECTION: {collection_name}")
    print(f"   Using {max_workers} worker threads")
    print("="*80)

    # 1. Ensure collection exists
    vector_size = embedding.get_vector_size()
    
    if reset_collection:
        # Check if exists via client directly or try/except
        # Qdrant client usually exposed in vector_db.client
        try:
            if vector_db.client.collection_exists(collection_name):
                print(f"⚠️ FORCE RESET: Deleting collection '{collection_name}'...")
                vector_db.client.delete_collection(collection_name)
        except Exception as e:
            print(f"⚠️ Error checking/deleting collection: {e}")

    vector_db.create_collection_if_not_exists(collection_name, vector_size)

    try:
        df = load_jsonl(file_path)
    except Exception as e:
        print(f"Skipping {collection_name} due to load error.")
        return

    # Check for 'id' column
    if "id" not in df.columns:
         print(f"❌ Error: 'id' column missing in {file_path}. Cannot deduplicate accurately.")
         return

    print("🔨 Preparing data...")
    # Pre-calculate deterministic UUIDs
    df["uuid"] = df["id"].apply(lambda x: str(uuid.uuid5(uuid.NAMESPACE_DNS, str(x))))
    
    # Pre-format text
    df["combine_text"] = df.apply(build_combine_row, axis=1)
    
    # Keep original index for metadata
    df["original_index"] = df.index

    # Clean check existing
    existing_ids = get_existing_ids(vector_db, collection_name)
    print(f"🔍 Documents đã tồn tại: {len(existing_ids)}")

    # Filter out existing
    original_count = len(df)
    df = df[~df["uuid"].isin(existing_ids)]
    skipped = original_count - len(df)
    print(f"⏩ Skipped (already exist): {skipped}")
    print(f"🔥 Remaining to insert: {len(df)}\n")

    if df.empty:
        print("✅ Nothing new to insert.")
        return

    # Batch Splitting
    records = df.to_dict("records")
    batches = []
    for i in range(0, len(records), batch_size):
        batch_records = records[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        batches.append((batch_records, batch_num))
        
    print(f"📦 Total Batches to process: {len(batches)}")

    total_inserted = 0
    errors = 0
    
    # Parallel Processing using ThreadPoolExecutor
    # Note: SentenceTransformers encode is often CPU bound but can release GIL for some ops.
    # Qdrant upsert is Network IO bound.
    # Threading helps overlap Encoding of Batch N+1 with Upload of Batch N.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Prepare future tasks
        future_to_batch = {
            executor.submit(
                process_batch_worker, 
                batch, 
                embedding, 
                sparse_model,
                vector_db, 
                collection_name, 
                num, 
                file_path
            ): num for batch, num in batches
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_batch):
            success, result, b_num = future.result()
            
            if success:
                count = result
                total_inserted += count
                if b_num % 10 == 0: # Print every 10 batches to avoid spam
                     print(f"✅ Batch {b_num} done ({count} docs)")
            else:
                error_msg = result
                print(f"❌ Error {error_msg}")
                errors += batch_size # Approx

    # SUMMARY
    print("\n📌 SUMMARY:", collection_name)
    print(f"   Tổng dòng gốc: {original_count}")
    print(f"   Inserted:      {total_inserted}")
    print(f"   Skipped:       {skipped}")
    print(f"   Errors (approx): {errors}")
    print("="*80 + "\n")


# ============================================================
# 6. MAIN: Insert 4 clusters
# ============================================================
def main():
    # Use config settings
    DB_TYPE = config.VECTOR_DB_CONFIG["type"]
    EMBEDDING_PROVIDER = config.EMBEDDING_MODEL_CONFIG.get("provider", "huggingface") 
    
    # Configurable performance params
    BATCH_SIZE = 256
    MAX_WORKERS = 4 # Number of parallel threads

    print("\n==============================")
    print(f"🚀 START INSERTING COLLECTIONS (Parallel)")
    print(f"   DB: {DB_TYPE}")
    print(f"   Provider: {EMBEDDING_PROVIDER}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Workers: {MAX_WORKERS}")
    
    # ⚠️ IMPORTANT: Set this to True to delete old collection and re-create with new Schema (Dense + Sparse)
    # Since we are upgrading schema, we MUST recreate.
    RESET_COLLECTION = False
    print(f"   Reset Collection: {RESET_COLLECTION}")
    print("==============================\n")

    vector_db = VectorDatabase(db_type=DB_TYPE)
    embedding = EmbeddingModel(provider=EMBEDDING_PROVIDER)
    
    # Initialize Sparse Model if config says so
    sparse_model = None
    if config.VECTOR_DB_CONFIG.get("sparse", False):
         sparse_model_name = config.SPARSE_CONFIG.get("model", "Qdrant/bm25")
         print(f"Initializing Sparse Model: {sparse_model_name}")
         sparse_model = SparseEmbeddingModel(provider="fastembed", model_name=sparse_model_name)

    # Iterate over COLLECTION_MAPPING from config
    for key, info in config.COLLECTION_MAPPING.items():
        collection_name = info["name"]
        file_path = str(info["corpus_file"]) 

        print(f"Checking: {key} -> {file_path}")

        if not os.path.exists(file_path):
            print(f"⚠️ File không tồn tại: {file_path}")
            continue

        process_single_file(
            vector_db=vector_db,
            embedding=embedding,
            sparse_model=sparse_model,
            file_path=file_path,
            collection_name=collection_name,
            batch_size=BATCH_SIZE,
            max_workers=MAX_WORKERS,
            reset_collection=RESET_COLLECTION
        )

    print("\n🎉 DONE — ALL COLLECTIONS PROCESSED!\n")


if __name__ == "__main__":
    main()
