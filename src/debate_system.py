"""Project 8 — Multi-Agent Debate System.

N proposer agents answer independently, a critic scores each proposal,
consensus picks a winner, and an aggregator synthesizes the final answer
with a confidence value.

Proposers reply with plain text. The critic replies with JSON:
{"scores": [{"index": 0, "score": 0.0-1.0, "critique": "..."}, ...]}.
"""

import json


class Debate:
    def __init__(self, proposers: list, critic, aggregator):
        """proposers: LLM clients; critic and aggregator: LLM clients."""
        self.proposers = proposers
        self.critic = critic
        self.aggregator = aggregator

    def run(self, question: str) -> dict:
        """Run one debate round.

        Requirements:
            - Every proposer is asked the question independently (its prompt
              must NOT contain other proposals).
            - The critic is called once; its prompt must contain ALL proposals;
              parse its scores.
            - Winner = highest score; ties break on LOWER index (deterministic).
            - confidence = winner_score - mean(other scores), clamped to
              [0.0, 1.0]. Single proposer -> confidence = winner_score.
            - The aggregator is called once with the question, the winning
              proposal, and the critic's critique of it; its text response is
              the final answer.
            - Return {"answer": <aggregator text>, "winner_index": int,
              "confidence": float, "proposals": [str, ...],
              "scores": [float, ...]}.
        """
        proposals = [proposer.complete(question) for proposer in self.proposers]
        critic_prompt = (
            "Question: "
            + question
            + "\n\nProposals:\n"
            + "\n".join(
                f"{index}: {proposal}" for index, proposal in enumerate(proposals)
            )
        )
        critic_response = self.critic.complete(critic_prompt)
        critique_data = (
            json.loads(critic_response)
            if isinstance(critic_response, str)
            else critic_response
        )
        scored = {item["index"]: item for item in critique_data["scores"]}
        scores = [scored[index]["score"] for index in range(len(proposals))]
        winner_index = max(range(len(scores)), key=lambda index: scores[index])
        winner_score = scores[winner_index]
        confidence = (
            winner_score
            if len(scores) == 1
            else winner_score
            - (
                sum(
                    scores[index]
                    for index in range(len(scores))
                    if index != winner_index
                )
                / (len(scores) - 1)
            )
        )
        confidence = max(0.0, min(1.0, confidence))
        winner_critique = scored[winner_index].get("critique", "")
        aggregator_prompt = (
            f"Question: {question}\nWinning proposal: {proposals[winner_index]}\n"
            f"Critique: {winner_critique}"
        )
        answer = self.aggregator.complete(aggregator_prompt)
        return {
            "answer": answer,
            "winner_index": winner_index,
            "confidence": confidence,
            "proposals": proposals,
            "scores": scores,
        }
