import json

metrics = {
    "ans_acc": [],
    "RLf": [],
    "RBllm": [],
    "RBalg": []
}

with open("mt-rag-benchmark-main/outputs/rag_eval_output.jsonl", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        m = row["metrics"]

        metrics["ans_acc"].append(m["idk_eval"][0])
        metrics["RLf"].append(m["RL_F_idk"][0])
        metrics["RBllm"].append(m["RB_llm_idk"][0])
        metrics["RBalg"].append(m["RB_agg_idk"][0])

final_result = {
    "Ans.Acc.": sum(metrics["ans_acc"]) / len(metrics["ans_acc"]),
    "RLf": sum(metrics["RLf"]) / len(metrics["RLf"]),
    "RBllm": sum(metrics["RBllm"]) / len(metrics["RBllm"]),
    "RBalg": sum(metrics["RBalg"]) / len(metrics["RBalg"]),
}

print("===== FINAL MODEL-LEVEL METRICS =====")
for k, v in final_result.items():
    print(f"{k}: {v:.4f}")
print("====================================")
