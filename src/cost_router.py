"""Project 6 — Cost-Aware Agent Router.

Route tasks to the cheapest capable model, keep a hard token budget,
exit early on confident cheap answers, and report cost analytics.

Each model config: {"client": <llm>, "cost_per_call": float,
"max_complexity": int}. Model clients reply with JSON:
{"answer": "...", "confidence": 0.0-1.0}.
"""

import json


class BudgetExceeded(Exception):
    pass


def estimate_complexity(task: str) -> int:
    """Deterministic complexity heuristic.

    Requirements (exactly):
        - base = number of whitespace-separated words
        - +10 for each of these keywords present (case-insensitive):
          "analyze", "compare", "architecture", "multi-step", "prove"
    """
    keywords = {"analyze", "compare", "architecture", "multi-step", "prove"}
    words = task.split()
    return len(words) + 10 * sum(keyword in task.lower() for keyword in keywords)


class CostRouter:
    def __init__(
        self, models: dict[str, dict], budget: float, confidence_exit: float = 0.8
    ):
        """models: name -> config. Must set up self.ledger: list of
        {"task": str, "model": str, "cost": float, "escalated": bool}."""
        self.models = models
        self.budget = budget
        self.confidence_exit = confidence_exit
        self.ledger = []

    def route(self, task: str) -> str:
        """Return the name of the CHEAPEST model whose max_complexity >=
        estimate_complexity(task). No capable model -> the most capable one."""
        complexity = estimate_complexity(task)
        capable = [
            (name, config)
            for name, config in self.models.items()
            if config["max_complexity"] >= complexity
        ]
        if capable:
            return min(capable, key=lambda item: item[1]["cost_per_call"])[0]
        return max(self.models.items(), key=lambda item: item[1]["max_complexity"])[0]

    def run_task(self, task: str) -> dict:
        """Execute one task.

        Requirements:
            - Spending another call's cost must never push total spend past
              the budget: raise BudgetExceeded BEFORE calling the model.
            - Call the routed model. If its confidence >= confidence_exit,
              return WITHOUT escalating (early exit).
            - Otherwise escalate ONCE to the most capable (highest
              max_complexity) model, budget permitting; mark escalated=True in
              the ledger entries.
            - Return {"answer": ..., "model": <final model>, "cost": <total
              cost of this task>}.
        """

        def spend() -> float:
            return sum(entry["cost"] for entry in self.ledger)

        def call(model_name: str, escalated: bool) -> dict:
            config = self.models[model_name]
            cost = config["cost_per_call"]
            if spend() + cost > self.budget:
                raise BudgetExceeded("Calling this model would exceed the budget")
            response = config["client"].complete(task)
            data = json.loads(response) if isinstance(response, str) else response
            self.ledger.append(
                {
                    "task": task,
                    "model": model_name,
                    "cost": cost,
                    "escalated": escalated,
                }
            )
            return data

        initial_model = self.route(task)
        initial = call(initial_model, escalated=False)
        initial_cost = self.models[initial_model]["cost_per_call"]
        if initial.get("confidence", 0.0) >= self.confidence_exit:
            return {
                "answer": initial.get("answer"),
                "model": initial_model,
                "cost": initial_cost,
            }

        capable_model = max(
            self.models.items(), key=lambda item: item[1]["max_complexity"]
        )[0]
        if capable_model == initial_model:
            return {
                "answer": initial.get("answer"),
                "model": initial_model,
                "cost": initial_cost,
            }
        try:
            escalated = call(capable_model, escalated=True)
        except BudgetExceeded:
            return {
                "answer": initial.get("answer"),
                "model": initial_model,
                "cost": initial_cost,
            }

        self.ledger[-2]["escalated"] = True
        return {
            "answer": escalated.get("answer"),
            "model": capable_model,
            "cost": initial_cost + self.models[capable_model]["cost_per_call"],
        }

    def analytics(self) -> dict:
        """{"total_cost": float, "calls": int, "by_model": {name: cost},
        "escalation_rate": fraction of tasks that escalated (0.0 if none)}."""
        by_model = {}
        for entry in self.ledger:
            by_model[entry["model"]] = by_model.get(entry["model"], 0.0) + entry["cost"]
        tasks = {entry["task"] for entry in self.ledger}
        escalated_tasks = {entry["task"] for entry in self.ledger if entry["escalated"]}
        return {
            "total_cost": sum(entry["cost"] for entry in self.ledger),
            "calls": len(self.ledger),
            "by_model": by_model,
            "escalation_rate": len(escalated_tasks) / len(tasks) if tasks else 0.0,
        }
