import json
import argparse
import sys
import os

# ============================================
# FIXED PATH SETUP
# ============================================

# Path thật của project RAG
PROJECT_ROOT = r"C:\Users\Admin\Desktop\code\NLP_project"
RAG_DIR = os.path.join(PROJECT_ROOT, "Products_RAG-main")

print("[DEBUG] PROJECT_ROOT =", PROJECT_ROOT)
print("[DEBUG] RAG_DIR =", RAG_DIR)

# Thêm RAG_DIR vào sys.path
if RAG_DIR not in sys.path:
    sys.path.insert(0, RAG_DIR)

# Import class RAGSystem
try:
    from rag import RAGSystem
    print("[DEBUG] Imported RAGSystem successfully!")
except Exception as e:
    print("❌ ERROR: Cannot import RAGSystem from rag.py")
    print("DETAIL:", e)
    sys.exit(1)

# ============================================
# Evaluation Script
# ============================================

def load_tasks(path, limit=None):
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            tasks.append(json.loads(line))
    return tasks


def generate_with_your_model(prompt: str, rag: RAGSystem) -> str:
    result = rag.query(user_query=prompt, top_k=5, messages=[])
    return result["answer"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print("\n========== RUNNING RAG GENERATION ==========\n")

    # Load RAG model
    rag = RAGSystem()

    # Load tasks
    tasks = load_tasks(args.input, args.limit)

    predictions = []
    for task in tasks:
        prompt = task["input"][-1]["text"]
        answer = generate_with_your_model(prompt, rag)

        predictions.append({
            "task_id": task["task_id"],
            "input": task["input"],
            "contexts": task["contexts"],
            "targets": task["targets"],
            "predictions": [{"text": answer}]
        })

    # Save output
    with open(args.output, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"✔ DONE! Wrote {len(predictions)} predictions → {args.output}")


if __name__ == "__main__":
    main()
