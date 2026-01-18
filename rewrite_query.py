import json
import argparse
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# =========================
# LOAD ENV
# =========================
load_dotenv()

# =========================
# CONFIG
# =========================
DEFAULT_INPUT = "rag_taskAC.jsonl"
DEFAULT_OUTPUT = "final_rewrite.jsonl"

STYLE = "reasoning-aware"   # "reasoning-aware" | "factual"

AZURE_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
API_VERSION = os.environ["OPENAI_API_VERSION"]
TEMPERATURE = 0.2

# =========================
# AZURE CLIENT
# =========================
client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=API_VERSION,
)

# =========================
# PROMPTS
# =========================
REASONING_PROMPT = """
Rewrite the LAST user message in the conversation for a STRICT context-aware retrieval task.

Rewrite strategy:
- Replace vague or implicit references in the last user message with clear, explicit descriptions.
- Incorporate ALL important information and knowledge from the entire conversation.
- If the conversation discusses how a decision is made (e.g., options, mechanisms, constraints, trade-offs, or ways of evaluating value or risk), these MUST be explicitly preserved in the rewritten question.
- Do NOT abstract concrete decision variables into general concepts.
- Do NOT optimize for brevity; preserving full decision context is more important than conciseness.

Requirements:
- Preserve the original meaning and intent of the last user message.
- Do NOT add new information.
- Do NOT answer the question.
- The rewritten question must be fully understandable on its own and reflect the full decision context.

Conversation:
{conversation}

Return ONLY the rewritten user question.
""".strip()


# =========================
# HELPERS
# =========================
def format_conversation(conversation):
    lines = []
    for turn in conversation:
        speaker = turn.get("speaker", "").upper()
        text = turn.get("text", "").strip()
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def rewrite_last_user_turn_llm(conversation):
    conversation_text = format_conversation(conversation)

    prompt = REASONING_PROMPT.format(conversation=conversation_text)

    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        temperature=TEMPERATURE,
        messages=[
            {
                "role": "system",
                "content": "You are a precise query rewriting assistant for information retrieval."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content.strip()

# =========================
# PIPELINE
# =========================
def process_file(input_path, output_path, limit=None):
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for idx, line in enumerate(fin):
            if limit is not None and idx >= limit:
                break

            item = json.loads(line)
            task_id = item.get("task_id")
            conversation = item.get("input", [])

            rewritten = rewrite_last_user_turn_llm(conversation)

            output_obj = {
                "_id": task_id,
                "text": f"|user|: {rewritten}"
            }

            fout.write(json.dumps(output_obj, ensure_ascii=False) + "\n")

# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser("Azure OpenAI query rewrite")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    process_file(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit
    )

if __name__ == "__main__":
    main()
