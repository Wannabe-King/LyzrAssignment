"""Project 4 — Memory-Enabled Conversational Agent.

Short-term rolling buffer + long-term recall with relevance scoring,
LLM-based compression of overflow, and cross-session persistence.

No real embeddings: relevance = deterministic token-overlap score
(|query words ∩ memory words| / |query words|), lowercase, whitespace-split.
"""

import json


class Memory:
    def __init__(self, llm, short_window: int = 4):
        """llm is used only by compress(). Must set up:
        - self.short_term: list of the most recent `short_window` turns
        - self.long_term: list of {"text": str, "source": "turn"|"summary"}
        """
        self.llm = llm
        self.short_window = short_window
        self.short_term = []
        self.long_term = []

    def add_turn(self, text: str) -> None:
        """Append a turn.

        Requirements:
            - self.short_term holds at most `short_window` turns (most recent,
              in order).
            - A turn evicted from the buffer is moved to long_term with
              source "turn" — nothing is ever silently dropped.
        """
        self.short_term.append(text)
        if len(self.short_term) > self.short_window:
            evicted = self.short_term.pop(0)
            self.long_term.append({"text": evicted, "source": "turn"})

    def relevance(self, query: str, text: str) -> float:
        """Token-overlap score as defined in the module docstring.
        Empty query -> 0.0."""
        if not query:
            return 0.0
        query_tokens = set(query.lower().split())
        if not query_tokens:
            return 0.0
        text_tokens = set(text.lower().split())
        overlap = query_tokens.intersection(text_tokens)
        return len(overlap) / len(query_tokens)

    def recall(self, query: str, k: int = 3) -> list[str]:
        """Top-k long_term texts by relevance, highest first.

        Requirements:
            - Entries scoring 0 are never returned.
            - Ties keep insertion order (stable).
        """
        scored = []
        for item in self.long_term:
            score = self.relevance(query, item["text"])
            if score > 0:
                scored.append((score, item["text"]))

        # Sort by score descending. Since sort is stable, insertion order is kept for ties.
        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for score, text in scored[:k]]

    def compress(self) -> None:
        """Summarize long_term "turn" entries into one "summary" entry.

        Requirements:
            - Call self.llm.complete() once with a prompt containing every
              long_term turn text; the response replaces those entries as one
              {"text": <response>, "source": "summary"}.
            - Existing "summary" entries are preserved (not re-compressed).
            - No-op (no LLM call) when there are no "turn" entries.
        """
        turns = [item["text"] for item in self.long_term if item["source"] == "turn"]
        if not turns:
            return

        prompt = "Summarize the following conversation turns:\n" + "\n".join(turns)
        summary_text = self.llm.complete(prompt)

        # Retain summaries, replace turns with the new consolidated summary
        new_long_term = [item for item in self.long_term if item["source"] == "summary"]
        new_long_term.append({"text": summary_text, "source": "summary"})
        self.long_term = new_long_term

    def save(self, path: str) -> None:
        """Persist short_term + long_term as JSON."""
        data = {
            "short_term": self.short_term,
            "long_term": self.long_term,
            "short_window": self.short_window,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, llm, path: str) -> "Memory":
        """Restore a Memory (same short_window semantics) from save()'s JSON."""
        with open(path, "r") as f:
            data = json.load(f)

        memory = cls(llm, short_window=data.get("short_window", 4))
        memory.short_term = data.get("short_term", [])
        memory.long_term = data.get("long_term", [])
        return memory
