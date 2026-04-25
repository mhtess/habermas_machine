"""Pre-flight cost estimation for the classroom Habermas Machine app.

Given the inputs about to be sent to a deliberation round (question, opinions,
optional critiques + previous winner, model choice, number of candidates), this
module produces a rough estimate of:

  - number of LLM calls
  - input + output token volume
  - approximate USD cost range

The estimate exists so the user can sanity-check the spend before kicking off
a long, expensive run (e.g. 178 participants ranking 4 long candidate
statements). It is intentionally conservative and approximate — pricing in the
table below changes frequently and may be stale.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


# Approximate Gemini API pricing in USD per 1M tokens. These figures change
# regularly; treat the resulting estimate as a ballpark, not a quote.
# Source: ai.google.dev pricing pages, late 2025.
MODEL_PRICING_USD_PER_M = {
    'gemini-2.5-flash-lite':  {'input': 0.10, 'output': 0.40},
    'gemini-2.5-flash':       {'input': 0.30, 'output': 2.50},
    'gemini-flash-latest':    {'input': 0.30, 'output': 2.50},
    'gemini-3-flash-preview': {'input': 0.30, 'output': 2.50},
    'gemini-2.5-pro':         {'input': 1.25, 'output': 10.00},
    'gemini-pro-latest':      {'input': 1.25, 'output': 10.00},
    'gemini-3-pro-preview':   {'input': 1.25, 'output': 10.00},
}
# Conservative fallback if the selected model isn't in the table.
_FALLBACK_PRICING = {'input': 1.25, 'output': 10.00}

# Tokenizer approximation: Gemini's tokenizer averages ~4 chars/token for
# English prose. Off by 10-20% in either direction is fine for a confirmation
# prompt — we bracket the result with low/high uncertainty bands.
_CHARS_PER_TOKEN = 4

# Fixed prompt boilerplate (instructions, format examples) that prepends every
# call. Measured empirically from the prompts in statement_model/cot_model.py
# and reward_model/cot_ranking_model.py.
_STATEMENT_PROMPT_OVERHEAD_TOKENS = 1800
_RANKING_PROMPT_OVERHEAD_TOKENS = 1500

# Output budgets. Statements are intentionally long-form (paper-style ~700-1000
# words including reasoning); ranking responses are mostly the arrow-notation
# answer plus a short rationale per candidate.
_STATEMENT_OUTPUT_TOKENS = 2000
_RANKING_OUTPUT_TOKENS = 800

# Uncertainty band on the final estimate.
_LOW_FACTOR = 0.7
_HIGH_FACTOR = 1.4


@dataclass(frozen=True)
class CostEstimate:
  """Estimated cost summary for a single deliberation round."""
  num_llm_calls: int
  input_tokens: int
  output_tokens: int
  cost_low_usd: float
  cost_mid_usd: float
  cost_high_usd: float
  pricing_known: bool   # False => model not in pricing table, used fallback
  is_critique: bool


def _count_tokens(text: str | None) -> int:
  if not text:
    return 0
  return max(1, len(text) // _CHARS_PER_TOKEN)


def _sum_tokens(texts: Sequence[str] | None) -> int:
  if not texts:
    return 0
  return sum(_count_tokens(t) for t in texts)


def estimate_cost(
    *,
    question: str,
    opinions: Sequence[str],
    num_candidates: int,
    model_name: str,
    previous_winner: str | None = None,
    critiques: Sequence[str] | None = None,
    avg_candidate_statement_tokens: int = 1300,
) -> CostEstimate:
  """Estimates LLM cost for one mediate() round.

  Args:
    question: The discussion question.
    opinions: One opinion per participant.
    num_candidates: How many candidate statements will be generated.
    model_name: The Gemini model name (must match aistudio_client keys).
    previous_winner: Winning statement from the prior round (critique mode).
    critiques: Per-participant critiques (critique mode). When provided,
      previous_winner should also be set.
    avg_candidate_statement_tokens: Assumed length of each candidate
      statement, used for sizing the per-citizen ranking prompt. The default
      (1300) corresponds to ~700-1000 words.

  Returns:
    CostEstimate with token counts and a low/mid/high USD range.
  """
  is_critique = bool(critiques)
  n = len(opinions)

  q_tok = _count_tokens(question)
  opinions_tok = _sum_tokens(opinions)
  winner_tok = _count_tokens(previous_winner) if is_critique else 0
  critiques_tok = _sum_tokens(critiques) if is_critique else 0
  avg_opinion_tok = (opinions_tok / n) if n else 0
  avg_critique_tok = (critiques_tok / n) if (is_critique and n) else 0

  # Statement-generation prompts include the question + all opinions
  # (+ winner + all critiques in the critique round) + a fixed instruction
  # block. There are num_candidates such calls per round.
  per_statement_input = (
      q_tok
      + opinions_tok
      + winner_tok
      + critiques_tok
      + _STATEMENT_PROMPT_OVERHEAD_TOKENS
  )
  statement_input_total = num_candidates * per_statement_input
  statement_output_total = num_candidates * _STATEMENT_OUTPUT_TOKENS

  # Per-citizen ranking prompts include the question + that citizen's own
  # opinion + every candidate statement (+ winner + that citizen's critique
  # in the critique round) + a fixed instruction block. There are n such
  # calls per round.
  per_ranking_input = (
      q_tok
      + avg_opinion_tok
      + num_candidates * avg_candidate_statement_tokens
      + (winner_tok + avg_critique_tok if is_critique else 0)
      + _RANKING_PROMPT_OVERHEAD_TOKENS
  )
  ranking_input_total = n * per_ranking_input
  ranking_output_total = n * _RANKING_OUTPUT_TOKENS

  total_input = int(statement_input_total + ranking_input_total)
  total_output = int(statement_output_total + ranking_output_total)

  pricing = MODEL_PRICING_USD_PER_M.get(model_name)
  pricing_known = pricing is not None
  if pricing is None:
    pricing = _FALLBACK_PRICING

  mid_cost = (
      total_input * pricing['input'] / 1_000_000
      + total_output * pricing['output'] / 1_000_000
  )

  return CostEstimate(
      num_llm_calls=num_candidates + n,
      input_tokens=total_input,
      output_tokens=total_output,
      cost_low_usd=mid_cost * _LOW_FACTOR,
      cost_mid_usd=mid_cost,
      cost_high_usd=mid_cost * _HIGH_FACTOR,
      pricing_known=pricing_known,
      is_critique=is_critique,
  )
