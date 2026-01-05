from typing import List, Dict


class Reflection:
    def __init__(self, llm_client, llm_model: str):
        self.llm_client = llm_client
        self.llm_model = llm_model

    def rewrite(self, messages: List[Dict], current_query: str) -> str:
        """
        Rewrite the current user query into a standalone question
        that fully captures the conversational context.
        """

        system_prompt = """You are a query rewriting assistant.

Your task is to rewrite the user's latest question into a standalone,
self-contained question that can be understood without the prior conversation.

Rules:
- Do NOT answer the question.
- Do NOT add new information.
- Preserve the user's original intent.
- If the question asks to elaborate (e.g. "nói thêm", "tell me more", "why"),
  expand it based on the immediately preceding topic.
- If the question is already self-contained, return it unchanged.
"""

        prompt_messages = [{"role": "system", "content": system_prompt}]

        # Include last N turns as context
        for msg in messages[-6:]:
            if msg["role"] in ("user", "assistant"):
                prompt_messages.append(
                    {
                        "role": msg["role"],
                        "content": msg["content"]
                    }
                )

        # Latest user query (explicit)
        prompt_messages.append(
            {
                "role": "user",
                "content": current_query
            }
        )

        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=prompt_messages,
            temperature=0.0,
            max_tokens=120,
        )

        rewritten = response.choices[0].message.content.strip()

        # Fallback safety
        if not rewritten or len(rewritten) < 5:
            return current_query

        return rewritten
