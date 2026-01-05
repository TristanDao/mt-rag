# response_generator.py
import os
import time
from openai import AzureOpenAI, RateLimitError
from prompt_templates import PROMPT_REGISTRY


class ResponseGenerator:
    def __init__(
        self,
        temperature: float = 0.0,
        max_tokens: int = 800,
        max_retries: int = 5,
    ):
        """
        Azure OpenAI Response Generator (SAFE VERSION)
        - Uses Azure DEPLOYMENT NAME
        - Retry + backoff
        - Never crashes on empty response
        """

        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("OPENAI_API_VERSION"),
        )

        # Azure yêu cầu DEPLOYMENT NAME
        self.model = os.getenv("AZURE_OPENAI_DEPLOYMENT")

        if not self.model:
            raise RuntimeError(
                "AZURE_OPENAI_DEPLOYMENT is not set. "
                "It must be the Azure deployment name."
            )

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def generate(
        self,
        prompt_type: str,
        query: str,
        context: str
    ) -> str:
        """
        Generate response from Azure OpenAI
        - ALWAYS returns string
        - NEVER raises due to empty LLM content
        """

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

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,          # Azure DEPLOYMENT
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                # ===== SAFE EXTRACTION =====
                if not response or not response.choices:
                    return ""

                msg = response.choices[0].message
                if not msg or not msg.content:
                    return ""

                return msg.content.strip()

            except RateLimitError:
                # Exponential backoff
                sleep_time = 2 ** attempt
                time.sleep(sleep_time)

            except Exception as e:
                # Không cho batch chết
                print("⚠️ [WARN] LLM generation failed:", repr(e))
                return ""

        # Retry exhausted
        return ""
