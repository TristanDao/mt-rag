import glob
import os

def merge_files():
    data_dir = "Products_RAG_main/data_retrieval"
    outfile = os.path.join(data_dir, "RAG_all_top30.jsonl")
    
    # Get all matching files
    files = glob.glob(os.path.join(data_dir, "RAG_*_top30.jsonl"))
    
    # Filter out the output file itself to avoid self-inclusion loop
    # Normalize paths for comparison
    files_to_merge = []
    outfile_abs = os.path.abspath(outfile)
    
    for f in files:
        if os.path.abspath(f) != outfile_abs:
            files_to_merge.append(f)
            
    print(f"Found {len(files_to_merge)} files to merge: {[os.path.basename(f) for f in files_to_merge]}")
    
    with open(outfile, 'w', encoding='utf-8') as fout:
        for input_file in files_to_merge:
            print(f"Merging {input_file}...")
            with open(input_file, 'r', encoding='utf-8') as fin:
                # Add a newline just in case the file doesn't end with one, 
                # but better to read content and ensure format.
                # Since these are JSONL, just simple concatenation works if specific lines are integrity clean.
                content = fin.read()
                if content:
                    fout.write(content)
                    if not content.endswith('\n'):
                        fout.write('\n')
                        
    print(f"Successfully merged into {outfile}")

if __name__ == "__main__":
    merge_files()
