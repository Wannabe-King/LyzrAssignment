"""Project 7 — Event-Triggered Automation Agent.

Consume webhook/queue events with idempotent execution, bounded retries with
exponential backoff, and a dead-letter queue. No LLM involved — this grades
production automation engineering.

An event is {"id": str, "type": str, "payload": dict}.
"""

import time


class EventProcessor:
    def __init__(self, handlers: dict, max_retries: int = 3, sleep=None):
        """handlers: event type -> callable(payload) -> result.
        sleep: injectable sleep function (tests pass a recorder; default
        time.sleep). Must set up:
            - self.processed: {event_id: result} of successful events
            - self.dead_letter: list of {"event": event, "error": str,
              "attempts": int}
        """
        self.handlers = dict(handlers)
        self.max_retries = max(0, max_retries)
        self.sleep = time.sleep if sleep is None else sleep
        self.processed = {}
        self.dead_letter = []
        self._dead_letter_ids = set()

    def _dead_letter(self, event, error, attempts):
        """Record one failed event, without creating duplicate DLQ entries."""
        event_id = event.get("id") if isinstance(event, dict) else None
        if event_id is not None and event_id in self._dead_letter_ids:
          return
        self.dead_letter.append({
          "event": event,
          "error": str(error),
          "attempts": attempts,
        })
        if event_id is not None:
          self._dead_letter_ids.add(event_id)

    def process(self, event: dict) -> dict:
        """Process one event.

        Requirements:
            - IDEMPOTENT: an event id seen before (success OR dead-lettered)
              returns {"status": "duplicate"} without invoking the handler.
            - Unknown event type -> straight to dead_letter (no retries),
              return {"status": "dead_letter"}.
            - Handler exceptions: retry up to max_retries additional attempts,
              calling self.sleep(2 ** attempt) between attempts (1, 2, 4...).
            - Success -> {"status": "ok", "result": ...} and record in
              self.processed.
            - Still failing after retries -> append to dead_letter with the
              LAST error string and total attempt count, return
              {"status": "dead_letter"}.
            - process() never raises.
        """
        try:
          if not isinstance(event, dict):
            self._dead_letter(event, "Invalid event", 0)
            return {"status": "dead_letter"}

          event_id = event.get("id")
          if event_id in self.processed or event_id in self._dead_letter_ids:
            return {"status": "duplicate"}

          event_type = event.get("type")
          handler = self.handlers.get(event_type)
          if handler is None:
            self._dead_letter(event, f"Unknown event type: {event_type}", 0)
            return {"status": "dead_letter"}

          payload = event.get("payload")
          for attempt in range(self.max_retries + 1):
            try:
              result = handler(payload)
              self.processed[event_id] = result
              return {"status": "ok", "result": result}
            except Exception as exc:
              if attempt == self.max_retries:
                self._dead_letter(event, exc, attempt + 1)
                return {"status": "dead_letter"}
              try:
                self.sleep(2 ** attempt)
              except Exception:
                # A failed delay must not let processing escape its
                # no-raise contract; proceed with the next attempt.
                pass
        except Exception as exc:
          # This also protects against malformed event mappings or unusual
          # user-supplied handler registries.
          try:
            self._dead_letter(event, exc, 0)
          except Exception:
            pass
          return {"status": "dead_letter"}

    def replay_dead_letter(self) -> int:
        """Retry every dead-lettered event once more through process()
        (idempotency must not block the replay). Return how many succeeded.
        Events that fail again remain dead-lettered exactly once (no dupes)."""
        recovered = 0
        # Work from a snapshot so newly dead-lettered failures are not replayed
        # again in this same call.
        for entry in list(self.dead_letter):
          event = entry["event"]
          event_id = event.get("id") if isinstance(event, dict) else None

          # Temporarily remove the old record and its idempotency marker.
          # `process` can then retry it normally; a fresh failure adds one
          # replacement entry, while success leaves no dead-letter entry.
          try:
            self.dead_letter.remove(entry)
          except ValueError:
            continue
          if event_id is not None:
            self._dead_letter_ids.discard(event_id)

          result = self.process(event)
          if result.get("status") == "ok":
            recovered += 1
        return recovered
