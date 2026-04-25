"""Pre-flight cost estimation for the classroom Habermas Machine app.

Given the inputs about to be sent to a deliberation round (question, opinions,
optional critiques + previous winner, model choice, number of candidates), this
module produces a rough estimate of:

  - number of LLM calls
  - input + output token volume
  - approximate USD cost range
  - approximate wall-clock runtime range

The estimate exists so the user can sanity-check the spend before kicking off
a long, expensive run (e.g. 178 participants ranking 4 long candidate
statements). It is intentionally conservative and approximate — pricing and
latency vary day to day and the tables below may be stale.
"""

from __future__ import annotations

import math
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


# Approximate per-call latency model. Each entry is:
#   - prefill_tps:  tokens-per-second the model ingests prompt at
#   - decode_tps:   tokens-per-second the model emits output at
#   - base_s:       fixed network + queueing overhead per call
# These are coarse rules of thumb observed in practice; the dialog brackets
# the estimate with a wide low/high band to absorb the variance.
MODEL_LATENCY = {
    'gemini-2.5-flash-lite':  {'prefill_tps': 80000, 'decode_tps': 200, 'base_s': 1.0},
    'gemini-2.5-flash':       {'prefill_tps': 50000, 'decode_tps': 150, 'base_s': 1.5},
    'gemini-flash-latest':    {'prefill_tps': 50000, 'decode_tps': 150, 'base_s': 1.5},
    'gemini-3-flash-preview': {'prefill_tps': 50000, 'decode_tps': 150, 'base_s': 1.5},
    'gemini-2.5-pro':         {'prefill_tps': 20000, 'decode_tps': 80,  'base_s': 2.0},
    'gemini-pro-latest':      {'prefill_tps': 20000, 'decode_tps': 80,  'base_s': 2.0},
    'gemini-3-pro-preview':   {'prefill_tps': 20000, 'decode_tps': 80,  'base_s': 2.0},
}
_FALLBACK_LATENCY = {'prefill_tps': 20000, 'decode_tps': 80, 'base_s': 2.0}

# Real-world wall-clock variance is huge — backoff retries, slow days, queue
# spikes. These factors are wider than the cost band on purpose.
_RUNTIME_LOW_FACTOR = 0.6
_RUNTIME_HIGH_FACTOR = 2.5


# Context-window size (max input tokens a single LLM call can accept) per
# model. The table is approximate and may go stale; check the live docs at
# https://ai.google.dev/gemini-api/docs/models when in doubt.
MODEL_CONTEXT_WINDOW_TOKENS = {
    'gemini-2.5-flash-lite':  1_000_000,
    'gemini-2.5-flash':       1_000_000,
    'gemini-flash-latest':    1_000_000,
    'gemini-3-flash-preview': 1_000_000,
    'gemini-2.5-pro':         1_000_000,
    'gemini-pro-latest':      1_000_000,
    'gemini-3-pro-preview':   1_000_000,
}
# Conservative fallback if the selected model isn't in the table.
_FALLBACK_CONTEXT_WINDOW = 128_000

# Tokenizer approximation: Gemini's tokenizer averages ~4 chars/token for
# English prose. Off by 10-20% in either direction is fine for a confirmation
# prompt — we bracket the result with low/high uncertainty bands.
_CHARS_PER_TOKEN = 4

# Fixed prompt boilerplate (instructions, format examples) that prepends every
# call. Measured empirically from the prompts in statement_model/cot_model.py
# and reward_model/cot_ranking_model.py.
_STATEMENT_PROMPT_OVERHEAD_TOKENS = 1800
_RANKING_PROMPT_OVERHEAD_TOKENS = 1500

# Output budgets for the *default* (succinct) mode: one substantial
# paragraph (~200-300 words) plus a short chain-of-thought reasoning block.
# Long-form output is opt-in via target_word_count and overrides this.
_STATEMENT_OUTPUT_TOKENS = 900
_RANKING_OUTPUT_TOKENS = 800

# Default assumed candidate-statement length in the ranking prompt (succinct
# mode). When target_word_count is set, the ranking prompt scales with it
# instead.
_DEFAULT_CANDIDATE_STATEMENT_TOKENS = 450

# Uncertainty band on the final estimate.
_LOW_FACTOR = 0.7
_HIGH_FACTOR = 1.4


@dataclass(frozen=True)
class CostEstimate:
  """Estimated cost + runtime summary for a single deliberation round."""
  num_llm_calls: int
  input_tokens: int                   # Total summed across ALL calls.
  output_tokens: int                  # Total summed across ALL calls.
  max_single_prompt_tokens: int       # Biggest single prompt sent to the LLM.
  context_window_tokens: int          # Selected model's per-call input limit.
  cost_low_usd: float
  cost_mid_usd: float
  cost_high_usd: float
  runtime_low_s: float
  runtime_mid_s: float
  runtime_high_s: float
  pricing_known: bool   # False => model not in pricing table, used fallback
  latency_known: bool   # False => model not in latency table, used fallback
  context_window_known: bool          # False => fell back to conservative.
  is_critique: bool

  @property
  def fits_in_context(self) -> bool:
    """Whether the biggest single prompt fits the selected model's window."""
    return self.max_single_prompt_tokens <= self.context_window_tokens

  @property
  def context_window_utilisation(self) -> float:
    """Fraction of the context window that the biggest single prompt uses."""
    if self.context_window_tokens <= 0:
      return 1.0
    return self.max_single_prompt_tokens / self.context_window_tokens


def _count_tokens(text: str | None) -> int:
  if not text:
    return 0
  return max(1, len(text) // _CHARS_PER_TOKEN)


def _sum_tokens(texts: Sequence[str] | None) -> int:
  if not texts:
    return 0
  return sum(_count_tokens(t) for t in texts)


def _per_call_latency_s(input_tokens: float, output_tokens: float,
                        latency: dict) -> float:
  """Time-to-completion of a single LLM call given prompt/output token counts."""
  return (
      latency['base_s']
      + input_tokens / latency['prefill_tps']
      + output_tokens / latency['decode_tps']
  )


def estimate_cost(
    *,
    question: str,
    opinions: Sequence[str],
    num_candidates: int,
    model_name: str,
    previous_winner: str | None = None,
    critiques: Sequence[str] | None = None,
    avg_candidate_statement_tokens: int = _DEFAULT_CANDIDATE_STATEMENT_TOKENS,
    max_concurrent_calls: int = 1,
    target_word_count: int | None = None,
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
      statement, used for sizing the per-citizen ranking prompt. Defaults
      to a succinct-paragraph length; overridden when target_word_count
      is set.
    max_concurrent_calls: Cap on simultaneous LLM calls during the ranking
      phase. Used only for runtime estimation — has no effect on cost.
    target_word_count: If set, scales the assumed statement-output length
      (and the candidate-statement length used for ranking-prompt sizing)
      so the cost estimate reflects long-form output mode.

  Returns:
    CostEstimate with token counts, USD range, and wall-clock runtime range.
  """
  is_critique = bool(critiques)
  n = len(opinions)

  q_tok = _count_tokens(question)
  opinions_tok = _sum_tokens(opinions)
  winner_tok = _count_tokens(previous_winner) if is_critique else 0
  critiques_tok = _sum_tokens(critiques) if is_critique else 0
  avg_opinion_tok = (opinions_tok / n) if n else 0
  avg_critique_tok = (critiques_tok / n) if (is_critique and n) else 0

  # In long-output mode, both the candidate statements themselves and the
  # output budget grow proportionally to the requested word count. Add ~500
  # tokens for the chain-of-thought reasoning section that always precedes
  # the statement.
  if target_word_count is not None:
    statement_output_tokens = int(target_word_count * 1.35) + 500
    candidate_statement_tokens = int(target_word_count * 1.35)
  else:
    statement_output_tokens = _STATEMENT_OUTPUT_TOKENS
    candidate_statement_tokens = avg_candidate_statement_tokens

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
  statement_output_total = num_candidates * statement_output_tokens

  # Per-citizen ranking prompts include the question + that citizen's own
  # opinion + every candidate statement (+ winner + that citizen's critique
  # in the critique round) + a fixed instruction block. There are n such
  # calls per round.
  per_ranking_input = (
      q_tok
      + avg_opinion_tok
      + num_candidates * candidate_statement_tokens
      + (winner_tok + avg_critique_tok if is_critique else 0)
      + _RANKING_PROMPT_OVERHEAD_TOKENS
  )
  ranking_input_total = n * per_ranking_input
  ranking_output_total = n * _RANKING_OUTPUT_TOKENS

  total_input = int(statement_input_total + ranking_input_total)
  total_output = int(statement_output_total + ranking_output_total)

  # The biggest single prompt the LLM will see — we report this separately
  # from total throughput because the per-call context window, not the
  # round-total token count, is what triggers context-overflow truncation.
  # Statement-generation prompts include all opinions, so they dominate.
  max_single_prompt_tokens = int(max(per_statement_input, per_ranking_input))

  context_window = MODEL_CONTEXT_WINDOW_TOKENS.get(model_name)
  context_window_known = context_window is not None
  if context_window is None:
    context_window = _FALLBACK_CONTEXT_WINDOW

  pricing = MODEL_PRICING_USD_PER_M.get(model_name)
  pricing_known = pricing is not None
  if pricing is None:
    pricing = _FALLBACK_PRICING

  mid_cost = (
      total_input * pricing['input'] / 1_000_000
      + total_output * pricing['output'] / 1_000_000
  )

  # Wall-clock estimate. Two phases:
  #   1) Statement generation runs serially in a Python for-loop in
  #      machine._generate_statements (num_candidates calls back-to-back).
  #   2) Per-citizen ranking is dispatched to a thread pool capped at
  #      max_concurrent_calls, so its time is ceil(n / pool) batches of one
  #      ranking-call latency.
  latency = MODEL_LATENCY.get(model_name)
  latency_known = latency is not None
  if latency is None:
    latency = _FALLBACK_LATENCY

  per_statement_call_s = _per_call_latency_s(
      per_statement_input, statement_output_tokens, latency
  )
  per_ranking_call_s = _per_call_latency_s(
      per_ranking_input, _RANKING_OUTPUT_TOKENS, latency
  )
  statement_phase_s = num_candidates * per_statement_call_s
  pool = max(1, max_concurrent_calls)
  ranking_batches = math.ceil(n / pool) if n > 0 else 0
  ranking_phase_s = ranking_batches * per_ranking_call_s
  mid_runtime_s = statement_phase_s + ranking_phase_s

  return CostEstimate(
      num_llm_calls=num_candidates + n,
      input_tokens=total_input,
      output_tokens=total_output,
      max_single_prompt_tokens=max_single_prompt_tokens,
      context_window_tokens=context_window,
      cost_low_usd=mid_cost * _LOW_FACTOR,
      cost_mid_usd=mid_cost,
      cost_high_usd=mid_cost * _HIGH_FACTOR,
      runtime_low_s=mid_runtime_s * _RUNTIME_LOW_FACTOR,
      runtime_mid_s=mid_runtime_s,
      runtime_high_s=mid_runtime_s * _RUNTIME_HIGH_FACTOR,
      pricing_known=pricing_known,
      latency_known=latency_known,
      context_window_known=context_window_known,
      is_critique=is_critique,
  )
