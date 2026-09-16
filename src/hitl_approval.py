"""Project 5 — Human-in-the-Loop Approval Agent.

Uncertainty detection -> pause -> request human input -> resume with validated
context, with a complete, ordered audit trail.

The LLM replies with JSON: {"answer": "...", "confidence": 0.0-1.0,
"action": "<name>"|null}.
"""


class UnknownTicket(Exception):
    pass


class ApprovalAgent:
    def __init__(
        self,
        llm,
        confidence_threshold: float = 0.75,
        protected_actions: set[str] = frozenset({"delete", "send_email", "refund"}),
    ):
        """Must set up:
        - self.audit_log: append-only list of events, each
          {"event": str, "detail": dict} — events in the order they happen.
        - internal storage for paused tickets.
        """
        self.llm = llm
        self.threshold = confidence_threshold
        self.protected_actions = set(protected_actions)
        self.audit_log = []
        self.pending_tickets = {}
        self._next_ticket_id = 1

    def handle(self, request: str) -> dict:
        """Process a request.

        Requirements:
            - Call the LLM once, parse its JSON.
            - Log {"event": "request", ...} first, always.
            - AUTO path: confidence >= threshold AND action not protected ->
              log "completed" and return
              {"status": "completed", "answer": ..., "ticket": None}.
            - PAUSE path: low confidence OR protected action -> create a
              ticket id, log "paused" with a "reason" of "low_confidence" or
              "protected_action", store the pending answer/action, and return
              {"status": "pending", "ticket": <id>, "reason": ...}.
              The protected action MUST NOT be considered executed.
        """
        import json

        self.audit_log.append({"event": "request", "detail": {"request": request}})

        response = self.llm.complete(request)
        data = json.loads(response) if isinstance(response, str) else response

        confidence = data.get("confidence", 0.0)
        action = data.get("action")
        answer = data.get("answer")

        is_protected = action in self.protected_actions
        low_confidence = confidence < self.threshold

        if not low_confidence and not is_protected:
            self.audit_log.append(
                {"event": "completed", "detail": {"answer": answer, "action": action}}
            )
            return {"status": "completed", "answer": answer, "ticket": None}

        ticket_id = str(self._next_ticket_id)
        self._next_ticket_id += 1

        reason = "protected_action" if is_protected else "low_confidence"
        self.pending_tickets[ticket_id] = data

        self.audit_log.append(
            {
                "event": "paused",
                "detail": {
                    "ticket": ticket_id,
                    "reason": reason,
                    "answer": answer,
                    "action": action,
                },
            }
        )

        return {"status": "pending", "ticket": ticket_id, "reason": reason}

    def resume(self, ticket: str, approved: bool, human_note: str = "") -> dict:
        """Resume a paused ticket with the human decision.

        Requirements:
            - Unknown/already-resolved ticket -> raise UnknownTicket.
            - Log "human_decision" (with approved + note), then:
              approved -> log "completed", return {"status": "completed",
              "answer": <pending answer>}.
              rejected -> log "aborted", return {"status": "aborted",
              "answer": None}.
            - A ticket can be resumed exactly once.
        """
        if ticket not in self.pending_tickets:
            raise UnknownTicket(f"Ticket {ticket} not found or already resolved")

        data = self.pending_tickets.pop(ticket)
        self.audit_log.append(
            {
                "event": "human_decision",
                "detail": {"ticket": ticket, "approved": approved, "note": human_note},
            }
        )

        if approved:
            self.audit_log.append(
                {
                    "event": "completed",
                    "detail": {
                        "ticket": ticket,
                        "answer": data.get("answer"),
                        "action": data.get("action"),
                    },
                }
            )
            return {"status": "completed", "answer": data.get("answer")}
        else:
            self.audit_log.append(
                {
                    "event": "aborted",
                    "detail": {"ticket": ticket, "reason": "human_rejection"},
                }
            )
            return {"status": "aborted", "answer": None}

    def audit_trail(self, ticket: str | None = None) -> list[dict]:
        """Full audit log, or only events whose detail carries this ticket."""
        if ticket is None:
            return list(self.audit_log)
        return [
            event for event in self.audit_log if event["detail"].get("ticket") == ticket
        ]
