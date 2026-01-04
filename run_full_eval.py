import subprocess
import os
import sys
import glob

# Configuration
PYTHON_EXEC = sys.executable
SCRIPT_FORMAT = "Products_RAG_main/format_input.py"
SCRIPT_MERGE = "Products_RAG_main/merge_results.py"
SCRIPT_EVAL = "mt-rag-benchmark-main/scripts/evaluation/run_retrieval_eval.py"

DATASETS = ["clapnq", "cloud", "fiqa", "govt"]
BASE_QUERY_PATH = "mt-rag-benchmark-main/human/retrieval_tasks"
OUTPUT_DIR = "Products_RAG_main/data_retrieval"

# Solve OMP Error #15
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_command(command):
    print(f"Executing: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(e)
        sys.exit(1)

def main():
    # 1. Run format_input.py for each dataset
    print("=== Step 1: Formatting Inputs ===")
    for dataset in DATASETS:
        # Construct paths
        # Assumption: query file is at <BASE>/<dataset>/<dataset>_rewrite.jsonl
        # If not found, try <dataset>.jsonl or look for jsonl in the dir
        query_file = os.path.join(BASE_QUERY_PATH, dataset, f"{dataset}_rewrite.jsonl")
        
        # Fallback search if specific rewrite file doesn't exist
        if not os.path.exists(query_file):
            print(f"Warning: {query_file} not found. Searching for .jsonl in {os.path.join(BASE_QUERY_PATH, dataset)}")
            files = glob.glob(os.path.join(BASE_QUERY_PATH, dataset, "*.jsonl"))
            if files:
                query_file = files[0] # Take the first one
                print(f"Using {query_file}")
            else:
                print(f"Error: No jsonl file found for {dataset}")
                continue

        output_file = os.path.join(OUTPUT_DIR, f"RAG_{dataset}_top30.jsonl")
        
        cmd = f'"{PYTHON_EXEC}" {SCRIPT_FORMAT} --queries_file "{query_file}" --output_file "{output_file}" --collection "{dataset}"'
        run_command(cmd)

    # 2. Run merge_results.py
    print("\n=== Step 2: Merging Results ===")
    cmd_merge = f'"{PYTHON_EXEC}" {SCRIPT_MERGE}'
    run_command(cmd_merge)

    # 3. Run evaluation
    print("\n=== Step 3: Running Evaluation ===")
    input_eval = os.path.join(OUTPUT_DIR, "RAG_all_top30.jsonl")
    output_eval = os.path.join("Products_RAG_main", "results", "retrieval_evaluation.json")
    
    # Ensure results dir exists
    os.makedirs(os.path.dirname(output_eval), exist_ok=True)
    
    cmd_eval = f'"{PYTHON_EXEC}" {SCRIPT_EVAL} --input_file "{input_eval}" --output_file "{output_eval}"'
    run_command(cmd_eval)

    print("\n✅ Full Evaluation Pipeline Completed!")

if __name__ == "__main__":
    main()
