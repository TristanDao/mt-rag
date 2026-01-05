import os
import sys
import json
import argparse
import importlib.util

# =========================================================
# FORCE OFFLINE MODE (PHẢI ĐẶT TRƯỚC KHI LOAD rag.py)
# =========================================================
os.environ["RAG_OFFLINE_MODE"] = "1"

# =========================================================
# COLLECTION CONFIG
# =========================================================
COLLECTION_MAPPING = {
    "clapnq": "mt-rag-clapnq-elser-512-100-20240503",
    "govt": "mt-rag-govt-elser-512-100-20240611",
    "fiqa": "mt-rag-fiqa-beir-elser-512-100-20240501",
    "cloud": "mt-rag-ibmcloud-elser-512-100-20240502",
}

# =========================================================
# PATH RESOLUTION (ABSOLUTE, SAFE)
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# scripts/evaluation -> mt-rag-benchmark-main
BENCHMARK_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..")
)

# mt-rag-benchmark-main -> PROJECT ROOT
PROJECT_ROOT = os.path.abspath(
    os.path.join(BENCHMARK_ROOT, "..")
)

# PROJECT ROOT -> Products_RAG_main
RAG_DIR = os.path.join(PROJECT_ROOT, "Products_RAG_main")

print("[DEBUG] PROJECT_ROOT =", PROJECT_ROOT)
print("[DEBUG] RAG_DIR      =", RAG_DIR)

rag_path = os.path.join(RAG_DIR, "rag.py")

if not os.path.exists(rag_path):
    print(f"❌ ERROR: rag.py not found at {rag_path}")
    sys.exit(1)

# Cho phép rag.py import các module cùng thư mục
if RAG_DIR not in sys.path:
    sys.path.insert(0, RAG_DIR)

# =========================================================
# SAFE LOAD rag.py (KHÔNG dùng import rag)
# =========================================================
spec = importlib.util.spec_from_file_location("rag", rag_path)
rag_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag_module)

RAGSystem = rag_module.RAGSystem
RAGMode = rag_module.RAGMode

print("[DEBUG] Loaded RAGSystem & RAGMode successfully")

# =========================================================
# HELPERS
# =========================================================
def load_tasks(path):
    """Load toàn bộ tasks từ file jsonl"""
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tasks.append(json.loads(line))
    return tasks


def generate_with_your_model(prompt, contexts, rag):
    """
    MT-RAG OFFLINE MODE
    - Không rewrite
    - Không retrieve
    - Không embedding
    - LUÔN trả về string
    """
    result = rag.query(
        user_query=prompt,
        mode=RAGMode.OFFLINE,
        provided_contexts=contexts
    )

    answer = result.get("answer")

    if answer is None:
        return ""

    if not isinstance(answer, str):
        return str(answer)

    return answer.strip()



# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--collection",
        required=True,
        choices=COLLECTION_MAPPING.keys(),
        help="Collection key: clapnq | govt | fiqa | cloud"
    )
    args = parser.parse_args()

    print("\n========== RUNNING MT-RAG OFFLINE GENERATION ==========\n")

    rag = RAGSystem()

    target_collection_name = COLLECTION_MAPPING[args.collection]

    # -----------------------------------------------------
    # 1️⃣ LOAD ALL TASKS
    # -----------------------------------------------------
    all_tasks = load_tasks(args.input)

    # -----------------------------------------------------
    # 2️⃣ FILTER BY COLLECTION
    # -----------------------------------------------------
    tasks = [
        t for t in all_tasks
        if t.get("Collection") == target_collection_name
    ]

    # -----------------------------------------------------
    # 3️⃣ APPLY LIMIT AFTER FILTER (QUAN TRỌNG)
    # -----------------------------------------------------
    if args.limit:
        tasks = tasks[:args.limit]

    print(f"[INFO] Running collection = {args.collection}")
    print(f"[INFO] Matched tasks     = {len(tasks)}")

    predictions = []

    for idx, task in enumerate(tasks, start=1):
        print(f"[RUNNING] {idx}/{len(tasks)}: {task['task_id']}")

        prompt = task["input"][-1]["text"]
        contexts = task["contexts"]

        answer = generate_with_your_model(
            prompt=prompt,
            contexts=contexts,
            rag=rag
        )

        predictions.append({
            "task_id": task["task_id"],
            "input": task["input"],
            "contexts": task["contexts"],
            "targets": task["targets"],
            "predictions": [{"text": answer}]
        })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n✔ DONE! Wrote {len(predictions)} predictions → {args.output}")


if __name__ == "__main__":
    main()
