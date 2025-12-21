import config
from vector_db import VectorDatabase

def main():
    # Load config
    db_type = config.VECTOR_DB_CONFIG["type"]
    print(f"⚠️  WARNING: This script will DELETE all collections defined in config.COLLECTION_MAPPING.")
    print(f"Target DB: {db_type}\n")
    
    # Init DB
    try:
        vector_db = VectorDatabase(db_type=db_type)
    except Exception as e:
        print(f"Error connecting to DB: {e}")
        return

    # Iterate and delete
    for key, info in config.COLLECTION_MAPPING.items():
        collection_name = info["name"]
        print(f"🗑️  Deleting collection: {collection_name} ... ", end="")
        
        try:
            if db_type == "qdrant":
                if vector_db.client.collection_exists(collection_name):
                    vector_db.client.delete_collection(collection_name)
                    print("✅ Deleted")
                else:
                    print("⚠️  Not found (Skipped)")
            
            elif db_type == "chromadb":
               vector_db.client.delete_collection(collection_name)
               print("✅ Deleted")
               
            elif db_type == "mongodb":
                db = vector_db.client["vector_db"]
                db.drop_collection(collection_name)
                print("✅ Dropped")
                
            else:
                 print(f"❌ delete not implemented for {db_type}")

        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n✨ Reset complete. You can now run insert_jsonl_clusters.py again.")

if __name__ == "__main__":
    main()
