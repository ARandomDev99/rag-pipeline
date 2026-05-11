import time
from typing import Any

from openai import APIStatusError, OpenAI, RateLimitError

from .config import CFG


class LLM:
    def __init__(self) -> None:
        if not CFG.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.client = OpenAI(
            base_url=CFG.openai_base_url,
            api_key=CFG.openai_api_key,
            default_headers={"HTTP-Referer": CFG.referer, "X-Title": CFG.title},
        )

    def _call(self, model: str | None, **kwargs: Any):
        primary = model or CFG.llm_model
        last_err: Exception | None = None
        for m in (primary, CFG.llm_fallback):
            for backoff in (1, 3, 7):
                try:
                    return self.client.chat.completions.create(model=m, **kwargs)
                except RateLimitError as e:
                    last_err = e
                    time.sleep(backoff)
                except APIStatusError as e:
                    if e.status_code in (429, 502, 503):
                        last_err = e
                        time.sleep(backoff)
                        continue
                    raise
        raise RuntimeError("LLM call failed after retries on primary and fallback") from last_err

    def generate(self, messages: list[dict], model: str | None = None) -> str:
        resp = self._call(model, messages=messages, temperature=0.0)
        return (resp.choices[0].message.content or "").strip()

    def generate_tool_step(
        self,
        messages: list[dict],
        tools: list[dict],
        model: str | None = None,
    ):
        resp = self._call(
            model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0,
        )
        return resp.choices[0].message
