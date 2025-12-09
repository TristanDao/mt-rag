"""
create_generation_predictions_rag.py
-------------------------------------------------
Script sinh predictions JSONL cho evaluation generation,
tương thích hoàn toàn với mt-rag-benchmark-main.

Script này:
 - đọc file task JSONL
 - lấy câu hỏi cuối của user
 - gọi RAGSystem thông qua generate_with_rag.py
 - tạo file output chứa trường "predictions"
"""

import json
from pathlib import Path
from typing import Dict, List, Any

# Import hàm generate_with_your_model từ file bridge
from generate_with_rag import generate_with_your_model


# ============================================================
# 1) Hàm lấy câu hỏi cuối cùng của user trong multi-turn input
# ============================================================

def extract_last_user_query(input_list: List[Dict[str, Any]]) -> str:
    """
    Tìm tin nhắn cuối cùng của 'user' trong danh sách input.

    Args:
        input_list: danh sách messages (history)

    Returns:
        text của user cuối cùng
    """
    for turn in reversed(input_list):
        if turn.get("speaker") == "user":
            return turn.get("text", "")
    raise ValueError("Không tìm thấy user turn trong input.")


# ============================================================
# 2) Hàm chính sinh predictions
# ============================================================

def create_generation_predictions(input_file: str, output_file: str):
    print(f"[INFO] Reading tasks from: {input_file}")

    total = 0
    error_count = 0

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:
            task = json.loads(line)
            total += 1

            try:
                # Lấy câu hỏi cuối
                last_user_question = extract_last_user_query(task.get("input", []))

                # Gọi RAGSystem để tạo câu trả lời
                answer_text = generate_with_your_model(last_user_question)

                # Gắn vào predictions
                task["predictions"] = [
                    {
                        "text": answer_text
                    }
                ]

            except Exception as e:
                print(f"[ERROR] Task {task.get('task_id')} failed: {e}")
                task["predictions"] = [{"text": ""}]
                error_count += 1

            fout.write(json.dumps(task, ensure_ascii=False) + "\n")

            if total % 20 == 0:
                print(f"[INFO] Processed {total} tasks...")

    print("\n=====================================")
    print("[DONE] Created predictions JSONL")
    print(f"Total tasks: {total}")
    print(f"Errors: {error_count}")
    print(f"Output file: {output_file}")
    print("=====================================\n")



# ============================================================
# 3) CLI Entry Point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tạo predictions JSONL bằng RAGSystem")
    parser.add_argument("--input", "-i", required=True, help="Input task file (.jsonl)")
    parser.add_argument("--output", "-o", required=True, help="Output predictions file (.jsonl)")

    args = parser.parse_args()

    create_generation_predictions(args.input, args.output)

    print("Chạy evaluate bằng lệnh:")
    print(f"python scripts/evaluation/run_generation_eval.py -i {args.output} -o {args.output.replace('.jsonl', '_eval.jsonl')} --provider openai --openai_key <KEY> --azure_host <HOST>")
