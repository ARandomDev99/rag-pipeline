import time

from openai import APIStatusError, OpenAI, RateLimitError

from .config import CFG


class LLM:
    def __init__(self) -> None:
        self.client = OpenAI(
            base_url=CFG.openai_base_url,
            api_key=CFG.openai_api_key,
            default_headers={"HTTP-Referer": CFG.referer, "X-Title": CFG.title},
        )

    def generate(self, messages: list[dict], model: str | None = None) -> str:
        primary = model or CFG.llm_model
        for attempt, m in enumerate([primary, CFG.llm_fallback]):
            for backoff in (1, 3, 7):
                try:
                    resp = self.client.chat.completions.create(
                        model=m,
                        messages=messages,
                        temperature=0.0,
                    )
                    return (resp.choices[0].message.content or "").strip()
                except RateLimitError:
                    time.sleep(backoff)
                except APIStatusError as e:
                    if e.status_code in (429, 502, 503):
                        time.sleep(backoff)
                        continue
                    raise
        raise RuntimeError("LLM call failed after retries on primary and fallback")