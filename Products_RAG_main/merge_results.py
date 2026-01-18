import os
import sys
import glob
from pathlib import Path

# Add parent directory to sys.path to allow imports from Products_RAG_main package
sys.path.insert(0, str(Path(__file__).parent.parent))

from Products_RAG_main.format_input import create_retrieval_input
import Products_RAG_main.config as config

def run_pipeline():
    # 1. Configuration
    collections = ["clapnq", "govt", "fiqa", "cloud"]
    
    # Define paths
    base_dir = Path(__file__).parent.parent
    retrieval_tasks_dir = base_dir / "mt-rag-benchmark-main" / "human" / "retrieval_tasks"
    data_retrieval_dir = base_dir / "Products_RAG_main" / "data_retrieval"
    
    # Create output dir if not exists
    os.makedirs(data_retrieval_dir, exist_ok=True)

    generated_files = []
    
    print(f"🚀 STARTING AUTO_RETRIEVAL PIPELINE FOR {len(collections)} COLLECTIONS")
    print(f"   Output Directory: {data_retrieval_dir}")
    top_k = config.RAG_CONFIG.get("top_k", 10)

    # 2. Run Retrieval Loop
    for col in collections:
        print(f"\n{'='*60}")
        print(f"▶️ PROCESSING: {col.upper()}")
        print(f"{'='*60}")
        
        # Path: mt-rag-benchmark-main/human/retrieval_tasks/{col}/{col}_rewrite.jsonl
        queries_file = retrieval_tasks_dir / col / f"{col}_rewrite.jsonl"
        
        # Output: Products_RAG_main/data_retrieval/RAG_{col}_top10.jsonl
        output_filename = f"RAG_{col}_top{top_k}.jsonl"
        output_file = data_retrieval_dir / output_filename
        
        if not queries_file.exists():
            print(f"❌ Query file not found: {queries_file}")
            continue
            
        try:
            create_retrieval_input(
                queries_file=str(queries_file),
                output_file=str(output_file),
                collection_key=col,
                top_k=top_k
            )
            generated_files.append(str(output_file))
        except Exception as e:
            print(f"❌ Error processing {col}: {e}")

    # 3. Merge Results
    print(f"\n{'='*60}")
    print(f"🔄 MERGING {len(generated_files)} FILES")
    print(f"{'='*60}")
    
    if not generated_files:
        print("⚠️ No files generated. Exiting.")
        return

    merge_output = data_retrieval_dir / f"RAG_all_top{top_k}.jsonl"
    
    try:
        with open(merge_output, 'w', encoding='utf-8') as outfile:
            for fname in generated_files:
                print(f"   + Adding {os.path.basename(fname)}...")
                with open(fname, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(content)
                    if content and not content.endswith('\n'):
                        outfile.write('\n')
                        
        print(f"\n✅ DONE! Final merged file: {merge_output}")
        print(f"   Size: {os.path.getsize(merge_output)} bytes")
        
    except Exception as e:
         print(f"❌ Error merging files: {e}")

if __name__ == "__main__":
    run_pipeline()
