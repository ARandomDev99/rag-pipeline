from dataclasses import dataclass, field

from .config import CFG


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class Memory:
    window_turns: int = CFG.memory_window_turns
    turns: list[Turn] = field(default_factory=list)

    def _trim(self) -> None:
        if 0 < self.window_turns < len(self.turns):
            del self.turns[: -self.window_turns]

    def add_user(self, content: str) -> None:
        self.turns.append(Turn("user", content))
        self._trim()

    def add_assistant(self, content: str) -> None:
        self.turns.append(Turn("assistant", content))
        self._trim()

    def history_messages(self) -> list[dict]:
        return [{"role": t.role, "content": t.content} for t in self.turns]

    def retrieval_hint(self, max_chars: int) -> str:
        # Concatenate the last user question and a snippet of the last
        # assistant reply so embedding picks up nouns elided by follow-ups
        # like "what about that?" or "tell me more about point 6".
        last_user = next(
            (t.content for t in reversed(self.turns) if t.role == "user"), ""
        )
        last_asst = next(
            (t.content for t in reversed(self.turns) if t.role == "assistant"), ""
        )
        snippet = last_asst[:max_chars]
        parts = [p for p in (last_user, snippet) if p]
        return "\n".join(parts)