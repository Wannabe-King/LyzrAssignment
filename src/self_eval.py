"""Project 9 — Self-Reflective Agent with Auto-Eval.

Execute -> LLM-as-judge -> critique -> regenerate under critique constraints,
keeping improvement metrics and returning the best attempt.

The worker LLM replies with plain text. The judge replies with JSON:
{"score": 0.0-1.0, "critique": "..."}.
"""
import json


class SelfEvalAgent:
    def __init__(self, worker, judge, pass_threshold: float = 0.8,
                 max_attempts: int = 3):
        """Must set up self.history: one entry per attempt,
        {"attempt": int (1-based), "output": str, "score": float,
        "critique": str}."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        self.worker = worker
        self.judge = judge
        self.pass_threshold = pass_threshold
        self.max_attempts = max_attempts
        self.history = []

    def run(self, task: str, criteria: str) -> dict:
        """Generate-judge-refine loop.

        Requirements:
            - Attempt 1: worker prompt contains the task and the criteria.
            - Judge every attempt: judge prompt must contain the criteria and
              the attempt's output; parse score + critique.
            - score >= pass_threshold -> stop immediately.
            - Otherwise regenerate: the next worker prompt MUST contain the
              judge's critique of the previous attempt (this is what makes it
              self-reflective) — plus the previous output.
            - Stop after max_attempts regardless.
            - Return {"output": <output of the BEST-scoring attempt>,
              "score": <its score>, "passed": bool,
              "attempts": <number made>,
              "improvement": <last score - first score, 0.0 for one attempt>}.
        """
        previous_output = None
        previous_critique = None
        attempts = []

        for attempt_number in range(1, self.max_attempts + 1):
            if previous_output is None:
                worker_prompt = (
                    f"Complete this task:\n{task}\n\n"
                    f"Success criteria:\n{criteria}"
                )
            else:
                worker_prompt = (
                    f"Complete this task:\n{task}\n\n"
                    f"Success criteria:\n{criteria}\n\n"
                    f"Previous output:\n{previous_output}\n\n"
                    f"Judge critique to address:\n{previous_critique}\n\n"
                    "Regenerate an improved response that addresses the critique."
                )

            output = self.worker.complete(worker_prompt)
            judge_prompt = (
                f"Evaluate the following output against these criteria:\n{criteria}\n\n"
                f"Output:\n{output}\n\n"
                'Return JSON with "score" (0.0-1.0) and "critique".'
            )
            judgment = json.loads(self.judge.complete(judge_prompt))
            score = float(judgment["score"])
            critique = str(judgment["critique"])

            record = {
                "attempt": attempt_number,
                "output": output,
                "score": score,
                "critique": critique,
            }
            self.history.append(record)
            attempts.append(record)

            if score >= self.pass_threshold:
                break

            previous_output = output
            previous_critique = critique

        best_attempt = max(attempts, key=lambda attempt: attempt["score"])
        improvement = 0.0 if len(attempts) == 1 else (
            attempts[-1]["score"] - attempts[0]["score"]
        )
        return {
            "output": best_attempt["output"],
            "score": best_attempt["score"],
            "passed": best_attempt["score"] >= self.pass_threshold,
            "attempts": len(attempts),
            "improvement": improvement,
        }
