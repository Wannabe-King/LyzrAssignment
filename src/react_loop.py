"""Project 2 — ReAct Planning Agent.

Observe -> think -> act loop with a hard iteration cap, unknown-tool recovery,
and graceful degradation instead of infinite looping.

The LLM is called with the running trace and must reply with JSON:
    {"thought": "...", "action": "<tool name>", "args": {...}}   -- act
    {"thought": "...", "final": "<answer>"}                       -- finish
"""

class ReActAgent:
    def __init__(self, llm, tools: dict, max_iterations: int = 5):
        """tools maps name -> callable(**args) -> str.

        Must set up self.trace: list of step dicts, in order, each
        {"thought": str, "action": str | None, "observation": str | None}.
        """
        self.trace=[]
        self.llm=llm
        self.tools=tools
        self.max_iter=max_iterations

    def run(self, goal: str) -> dict:
        """Run the ReAct loop for a goal.

        Requirements:
            - Each iteration: call the LLM with the goal plus all prior
              thoughts/observations, parse its JSON decision.
            - "final" decision -> return {"status": "done", "answer": final,
              "iterations": n}.
            - Action decision -> execute the tool, append the observation to
              the trace, continue.
            - Unknown tool or tool exception MUST NOT crash the loop: record
              an error observation ("ERROR: ...") and continue, letting the
              model recover.
            - Unparseable LLM output counts as an iteration with observation
              "ERROR: invalid decision format".
            - After max_iterations without "final": degrade gracefully -->
              return {"status": "max_iterations", "answer": <best-effort
              summary of the trace, non-empty>, "iterations": max_iterations}.
              Never loop past the cap; never raise.
        """
        import json

        for i in range(self.max_iter):
            # Construction of the prompt including the trace
            history = ""
            for step in self.trace:
                history += f"Thought: {step['thought']}\n"
                if step['action']:
                    history += f"Action: {step['action']}\nObservation: {step['observation']}\n"
            
            prompt = f"Goal: {goal}\n{history}\nDecision (JSON):"
            
            try:
                raw_response=self.llm.complete(prompt)
                decision=json.loads(raw_response)
                thought=decision.get("thought", "")
                
                if "final" in decision:
                    self.trace.append({"thought": thought, "action": None, "observation": None})
                    return {
                        "status": "done",
                        "answer": decision["final"],
                        "iterations": i + 1
                    }
                
                action = decision.get("action")
                args = decision.get("args", {})
                
                if action not in self.tools:
                    observation = f"ERROR: unknown tool {action}"
                else:
                    try:
                        observation = self.tools[action](**args)
                    except Exception as e:
                        observation = f"ERROR: {e}"
                
                self.trace.append({
                    "thought": thought,
                    "action": action,
                    "observation": observation
                })

            except (json.JSONDecodeError, Exception):
                self.trace.append({
                    "thought": "Invalid or unparseable response",
                    "action": None,
                    "observation": "ERROR: invalid decision format"
                })

        # Graceful degradation after max iterations
        summary = "Cap reached. Steps taken: " + "; ".join(
            [f"{t['thought']} ({t['action'] or 'no action'})" for t in self.trace]
        )
        return {
            "status": "max_iterations",
            "answer": summary,
            "iterations": self.max_iter
        }