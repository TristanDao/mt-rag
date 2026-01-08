from typing import List, Dict


class Reflection:
    """
    Reflection = USER-SPEECH NORMALIZER

    Responsibilities:
    - Rewrite the user's input so it is self-contained for retrieval.
    - Preserve the user's identity, intent, tone, and sentence type.
    - Use prior USER messages as conversational memory.
    - Resolve vague references when the referent is unambiguous.
    - NEVER answer, execute, explain, or role-play.
    """

    def __init__(self, llm_client, llm_model: str):
        self.llm_client = llm_client
        self.llm_model = llm_model

    # =====================================================
    # INTERNAL: BUILD CONVERSATION MEMORY
    # =====================================================
    def _build_memory(self, messages: List[Dict], max_items: int = 5) -> str:
        """
        Build a lightweight conversation memory summary
        from prior USER messages only.

        This is NOT a summary of dialogue,
        but a condensation of stable facts, goals, and constraints.
        """

        memory_items = []

        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue

            content = msg.get("content", "").strip()
            if not content:
                continue

            # Heuristic: keep messages that look like
            # definitions, decisions, constraints, or corrections
            if (
                len(content) > 40
                or any(
                    kw in content.lower()
                    for kw in [
                        "tôi muốn",
                        "tôi không",
                        "ý tôi",
                        "tôi đang",
                        "tôi dùng",
                        "tôi không dùng",
                        "yêu cầu",
                        "mục tiêu",
                        "hệ thống",
                        "rag",
                        "prompt",
                        "rewrite",
                    ]
                )
            ):
                memory_items.append(content)

            if len(memory_items) >= max_items:
                break

        if not memory_items:
            return ""

        memory_items.reverse()
        return "\n".join(f"- {item}" for item in memory_items)

    # =====================================================
    # CORE REWRITE
    # =====================================================
    def rewrite(self, messages: List[Dict], current_query: str) -> str:
        """
        Rewrite the user's input into a self-contained, natural form
        suitable for retrieval, using prior USER messages as memory.

        If rewrite fails, return the original input.
        """

        system_prompt = """
You are a USER-SPEECH NORMALIZATION TOOL.

Your task is to rewrite the user's CURRENT input so that it is self-contained
ONLY when the missing reference can be resolved with certainty
from prior USER messages.

CRITICAL RULES:
- NEVER invent placeholders (e.g. "your name", "[name]", "the user").
- NEVER generalize or anonymize identity references.
- If a reference (e.g. "my name") cannot be resolved with certainty,

IDENTITY:
- "I", "me", "my", "tôi", "mình" ALWAYS refer to the same human user.
- You are NOT an assistant.
- You are NOT addressing anyone.

REWRITE POLICY:
- Rewrite ONLY when it increases clarity WITHOUT losing information.
- If rewrite would remove or abstract identity information, DO NOT rewrite.
- Preserve sentence type (question, request, statement).

ABSOLUTELY FORBIDDEN:
- Adding placeholders.
- Adding brackets.
- Adding explanations.
- Answering the request.

OUTPUT:
- Output ONLY the rewritten user input.
"""

        prompt_messages = [
            {"role": "system", "content": system_prompt}
        ]

        # -------------------------------------------------
        # Inject conversation memory (if any)
        # -------------------------------------------------
        memory = self._build_memory(messages)
        if memory:
            prompt_messages.append(
                {
                    "role": "system",
                    "content": f"Conversation memory (from the same user):\n{memory}"
                }
            )

        # -------------------------------------------------
        # Append current input ONCE
        # -------------------------------------------------
        prompt_messages.append(
            {
                "role": "user",
                "content": current_query
            }
        )

        # -------------------------------------------------
        # Call LLM with HARD FAIL-SAFE
        # -------------------------------------------------
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=prompt_messages,
                temperature=0.0,
                max_tokens=120,
                timeout=10,
            )

            rewritten = response.choices[0].message.content.strip()

            return rewritten or current_query

        except Exception as e:
            print(f"[Reflection] Rewrite failed: {e}")
            return current_query
