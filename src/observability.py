"""Project 10 — Production Agent Observability.

Instrument an agent with tracing spans, per-call cost/latency accounting,
and a repeated-prompt loop alarm — the testable core of what LangSmith/Arize
give you in production.
"""

import time
from contextlib import contextmanager


class Tracer:
    def __init__(self, clock=None):
        """clock: injectable time function (tests pass a fake; default
        time.monotonic). Must set up self.spans: list of finished spans,
        {"name": str, "duration": float, "parent": str | None}, appended in
        FINISH order."""
        self.clock = clock or time.monotonic
        self.spans = []
        self._active_spans = []

    def span(self, name: str):
        """Context manager measuring a named span.

        Requirements:
            - duration = clock() at exit - clock() at entry.
            - Nested spans record their enclosing span's name as parent
              (top level -> None).
            - The span is recorded even if the body raises (exception must
              propagate).
        """

        @contextmanager
        def tracked_span():
            parent = self._active_spans[-1] if self._active_spans else None
            started = self.clock()
            self._active_spans.append(name)
            try:
                yield
            finally:
                self._active_spans.pop()
                self.spans.append(
                    {"name": name, "duration": self.clock() - started, "parent": parent}
                )

        return tracked_span()


class InstrumentedLLM:
    """Wrap an LLM client with cost accounting and a loop alarm."""

    def __init__(
        self, llm, tracer: Tracer, cost_per_call: float = 0.01, loop_threshold: int = 3
    ):
        """Must set up:
        - self.total_cost, self.call_count
        - self.alerts: list of {"type": "loop", "prompt": str,
          "count": int}
        """
        self.llm = llm
        self.tracer = tracer
        self.cost_per_call = cost_per_call
        self.loop_threshold = loop_threshold
        self.total_cost = 0.0
        self.call_count = 0
        self.alerts = []
        self._prompt_counts = {}

    def complete(self, prompt: str) -> str:
        """Delegate to the wrapped llm inside a tracer span named "llm.complete".

        Requirements:
            - Add cost_per_call to total_cost per call (also on failures).
            - Loop alarm: when the SAME prompt string is seen for the
              loop_threshold-th time, append one alert (once per threshold
              multiple: at 3, 6, 9... for threshold 3).
        """
        self.call_count += 1
        self.total_cost += self.cost_per_call
        count = self._prompt_counts.get(prompt, 0) + 1
        self._prompt_counts[prompt] = count
        if count % self.loop_threshold == 0:
            self.alerts.append({"type": "loop", "prompt": prompt, "count": count})
        with self.tracer.span("llm.complete"):
            return self.llm.complete(prompt)

    def report(self) -> dict:
        """{"calls": int, "total_cost": float, "avg_latency": float
        (mean duration of llm.complete spans, 0.0 when none),
        "alerts": <the alerts list>}."""
        durations = [
            span["duration"]
            for span in self.tracer.spans
            if span["name"] == "llm.complete"
        ]
        return {
            "calls": self.call_count,
            "total_cost": self.total_cost,
            "avg_latency": sum(durations) / len(durations) if durations else 0.0,
            "alerts": self.alerts,
        }
