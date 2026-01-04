# response_generator.py

from openai import OpenAI
from prompt_templates import PROMPT_REGISTRY


class ResponseGenerator:
    def __init__(
        self,
        llm_client: OpenAI,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 800
    ):
        self.client = llm_client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(
        self,
        prompt_type: str,
        query: str,
        context: str
    ) -> str:

        if prompt_type not in PROMPT_REGISTRY:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

        prompt_cfg = PROMPT_REGISTRY[prompt_type]

        messages = [
            {
                "role": "system",
                "content": prompt_cfg["system"]
            },
            {
                "role": "user",
                "content": prompt_cfg["user"].format(
                    context=context,
                    query=query
                )
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return response.choices[0].message.content.strip()
