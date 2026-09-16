"""Project 3 — Multi-Tool Orchestrator.

Dynamic tool registry, capability-based routing with priority conflict
resolution, permission scoping, and parallel execution.
"""
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Tool:
    name: str
    fn: Callable
    capabilities: set[str]
    required_scope: str | None = None  # None = public
    priority: int = 0                  # higher wins when capabilities conflict


class PermissionDenied(Exception):
    pass


class Orchestrator:
    def __init__(self):
        """Set up an empty registry."""
        self.registry = {}

    def register(self, tool: Tool) -> None:
        """Add a tool. Re-registering the same name replaces it."""
        self.registry[tool.name] = tool

    def resolve(self, capability: str) -> Tool:
        """Return the tool for a capability.

        Requirements:
            - If several tools share the capability, the highest `priority`
              wins; ties break alphabetically by name (deterministic).
            - Unknown capability -> KeyError.
        """
        candidates = [
            t for t in self.registry.values() if capability in t.capabilities
        ]
        if not candidates:
            raise KeyError(f"No tool found for capability: {capability}")
        
        # Sort by priority descending, then by name ascending
        candidates.sort(key=lambda t: (-t.priority, t.name))
        return candidates[0]

    def execute(self, capability: str, scopes: set[str], **kwargs):
        """Resolve and run one tool.

        Requirements:
            - If the tool has a required_scope not present in `scopes`,
              raise PermissionDenied WITHOUT executing the tool.
        """
        tool = self.resolve(capability)
        if tool.required_scope and tool.required_scope not in scopes:
            raise PermissionDenied(f"Scope '{tool.required_scope}' required")
        return tool.fn(**kwargs)

    def execute_parallel(self, tasks: list[dict], scopes: set[str]) -> list[dict]:
        """Run many tasks concurrently (threads); each task is
        {"capability": str, "kwargs": dict}.

        Requirements:
            - MUST use real concurrency (concurrent.futures) — grading asserts
              wall-clock time of parallel sleeps.
            - Results return IN INPUT ORDER as
              {"ok": True, "result": ...} or {"ok": False, "error": str}.
            - One failing/forbidden task must not affect the others.
        """
        from concurrent.futures import ThreadPoolExecutor
        def run_task(task: dict) -> dict:
            try:
                result = self.execute(
                    task["capability"],
                    scopes=scopes,
                    **task.get("kwargs", {}),
                )
                return {"ok": True, "result": result}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(run_task, task) for task in tasks]
            return [future.result() for future in futures]