import os
import uuid
import pandas as pd
from typing import Set

from embedding import EmbeddingModel
from vector_db import VectorDatabase


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
# 3. Lấy danh sách title đã tồn tại
# ============================================================
def get_existing_titles(vector_db: VectorDatabase, collection_name: str) -> Set[str]:
    existing_titles = set()

    try:
        if vector_db.db_type == "mongodb":
            db = vector_db.client.get_database("vector_db")
            collection = db[collection_name]
            docs = collection.find({}, {"title": 1})
            existing_titles = {d["title"] for d in docs}

        elif vector_db.db_type == "chromadb":
            collection = vector_db.client.get_collection(name=collection_name)
            results = collection.get()
            if results and "metadatas" in results:
                existing_titles = {m.get("title") for m in results["metadatas"]}

        elif vector_db.db_type == "qdrant":
            scroll_result = vector_db.client.scroll(
                collection_name=collection_name,
                limit=10000,
                with_payload=True
            )
            for p in scroll_result[0]:
                if "title" in p.payload:
                    existing_titles.add(p.payload["title"])

        elif vector_db.db_type == "supabase":
            resp = vector_db.client.table(collection_name).select("title").execute()
            existing_titles = {r["title"] for r in resp.data}

    except Exception as e:
        print(f"⚠️ Note: {e}")

    return existing_titles


# ============================================================
# 4. Insert batch
# ============================================================
def insert_batch(vector_db: VectorDatabase, batch_data: list, collection_name: str):
    try:
        vector_db.insert(data=batch_data, collection_name=collection_name)
        return True, None
    except Exception as e:
        return False, str(e)


# ============================================================
# 5. Xử lý 1 file → 1 collection
# ============================================================
def process_single_file(vector_db, embedding, file_path, collection_name, batch_size=50):
    print("\n" + "="*80)
    print(f"🚀 PROCESSING COLLECTION: {collection_name}")
    print("="*80)

    df = load_jsonl(file_path)
    df["information"] = df.apply(build_combine_row, axis=1)


    existing_titles = get_existing_titles(vector_db, collection_name)
    print(f"🔍 Documents đã tồn tại: {len(existing_titles)}\n")

    inserted = 0
    skipped = 0
    errors = 0
    batch = []
    batch_num = 1

    for index, row in df.iterrows():
        title = row["title"]
        text = row["information"]

        if title in existing_titles:
            skipped += 1
            continue

        try:
            vec = embedding.encode_single(text)
            doc_id = str(uuid.uuid4())

            batch.append({
                "id": doc_id,
                "title": title,
                "embedding": vec,
                "information": text,
                "metadata": {
                    "title": title,
                    "index": index,
                    "source": os.path.basename(file_path)
                }
            })

            if len(batch) >= batch_size:
                ok, err = insert_batch(vector_db, batch, collection_name)

                if ok:
                    inserted += len(batch)
                    print(f"Batch {batch_num} → Inserted {len(batch)}")
                else:
                    errors += len(batch)
                    print(f"❌ Batch {batch_num}: {err}")

                batch = []
                batch_num += 1

        except Exception as e:
            print(f"❌ Lỗi xử lý '{title}': {e}")
            errors += 1

    # Insert batch cuối
    if batch:
        ok, err = insert_batch(vector_db, batch, collection_name)
        if ok:
            inserted += len(batch)
            print(f"Batch cuối → Inserted {len(batch)}")
        else:
            errors += len(batch)

    # SUMMARY
    print("\n📌 SUMMARY:", collection_name)
    print(f"   Tổng dòng: {len(df)}")
    print(f"   Inserted: {inserted}")
    print(f"   Skipped:  {skipped}")
    print(f"   Errors:   {errors}")
    print("="*80 + "\n")


# ============================================================
# 6. MAIN: Insert 4 clusters
# ============================================================
def main():
    DATA_FOLDER = "datasets/"
    DB_TYPE = "mongodb"
    EMBEDDING_PROVIDER = "huggingface"
    BATCH_SIZE = 10000

    FILE_CLUSTERS = {
        "clapnq.jsonl": "clapnq",
        "cloud.jsonl": "cloud",
        "fiqa.jsonl": "fiqa",
        "govt.jsonl": "govt"
    }

    print("\n==============================")
    print("🚀 START INSERTING 4 CLUSTERS")
    print("==============================\n")

    vector_db = VectorDatabase(db_type=DB_TYPE)
    embedding = EmbeddingModel(provider=EMBEDDING_PROVIDER)

    for filename, collection in FILE_CLUSTERS.items():
        file_path = os.path.join(DATA_FOLDER, filename)

        if not os.path.exists(file_path):
            print(f"⚠️ File không tồn tại: {file_path}")
            continue

        process_single_file(
            vector_db=vector_db,
            embedding=embedding,
            file_path=file_path,
            collection_name=collection,
            batch_size=BATCH_SIZE
        )

    print("\n🎉 DONE — ALL 4 COLLECTIONS INSERTED SUCCESSFULLY!\n")


if __name__ == "__main__":
    main()
