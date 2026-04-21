"""End-to-end smoke test with per-stage timing for the Habermas Machine.

Runs one (and optionally two) rounds of `HabermasMachine.mediate()` and prints
a timing breakdown for statement generation, ranking, and the rest.

Usage:
  # Logic-only, no API key needed, runs in < 1 s.
  python smoke_test.py --mock

  # Real Gemini API (requires GOOGLE_API_KEY, costs a few cents).
  python smoke_test.py

  # Smaller/faster real run for iteration.
  python smoke_test.py --num-candidates 2 --num-citizens 2 --skip-critique
"""

import argparse
import os
import statistics
import sys
import time
from collections.abc import Collection

from habermas_machine import machine
from habermas_machine import types
from habermas_machine.llm_client import base_client
from habermas_machine.social_choice import utils as sc_utils


class TimingLLMClient(base_client.LLMClient):
  """Wraps an LLMClient and records per-call duration."""

  def __init__(self, wrapped: base_client.LLMClient):
    self._wrapped = wrapped
    self.calls: list[dict] = []

  def reset(self) -> None:
    self.calls = []

  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = base_client.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = base_client.DEFAULT_TERMINATORS,
      temperature: float = base_client.DEFAULT_TEMPERATURE,
      timeout: float = base_client.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    start = time.perf_counter()
    response = self._wrapped.sample_text(
        prompt,
        max_tokens=max_tokens,
        terminators=terminators,
        temperature=temperature,
        timeout=timeout,
        seed=seed,
    )
    duration = time.perf_counter() - start
    self.calls.append({
        'duration_s': duration,
        'prompt_chars': len(prompt),
        'response_chars': len(response),
    })
    return response


def _fmt_stats(label: str, durations: list[float]) -> str:
  if not durations:
    return f'  {label:<12} n=0   (no calls)'
  total = sum(durations)
  return (
      f'  {label:<12} n={len(durations):<3}'
      f' total {total:6.2f}s'
      f'  min {min(durations):5.2f}s'
      f'  mean {statistics.mean(durations):5.2f}s'
      f'  max {max(durations):5.2f}s'
  )


def run_round(
    hm: machine.HabermasMachine,
    round_label: str,
    inputs: list[str],
    statement_client: TimingLLMClient,
    reward_client: TimingLLMClient,
    num_candidates: int,
    num_citizens: int,
) -> dict:
  """Runs one mediation round and returns timing + correctness info."""
  statement_client.reset()
  reward_client.reset()

  start = time.perf_counter()
  winner, sorted_statements = hm.mediate(inputs)
  total = time.perf_counter() - start

  # The statement client only handles statement-generation calls; the reward
  # client only handles ranking calls. Each call may include retries, so we
  # don't assume exact counts — just report what actually happened.
  statement_durations = [c['duration_s'] for c in statement_client.calls]
  ranking_durations = [c['duration_s'] for c in reward_client.calls]

  # Correctness checks.
  checks = []
  checks.append(
      ('winner in sorted_statements', winner in sorted_statements)
  )
  checks.append(
      (
          f'len(sorted_statements) == {num_candidates}',
          len(sorted_statements) == num_candidates,
      )
  )

  llm_total = sum(statement_durations) + sum(ranking_durations)
  other = max(0.0, total - llm_total)

  return {
      'label': round_label,
      'total_s': total,
      'statement_durations': statement_durations,
      'ranking_durations': ranking_durations,
      'other_s': other,
      'winner': winner,
      'num_candidates': num_candidates,
      'num_citizens': num_citizens,
      'checks': checks,
  }


def print_summary(result: dict) -> None:
  print(f'\n{result["label"]}  —  total {result["total_s"]:.2f}s')
  print(_fmt_stats('statements', result['statement_durations']))
  print(_fmt_stats('ranking', result['ranking_durations']))
  print(f'  other        {result["other_s"]:6.2f}s'
        f'  (social choice + bookkeeping)')
  winner_preview = result['winner'][:120].replace('\n', ' ')
  if len(result['winner']) > 120:
    winner_preview += '...'
  print(f'  winner: {winner_preview}')
  for name, ok in result['checks']:
    mark = 'PASS' if ok else 'FAIL'
    print(f'  [{mark}] {name}')


def build_machine(
    args: argparse.Namespace,
    question: str,
) -> tuple[machine.HabermasMachine, TimingLLMClient, TimingLLMClient]:
  """Constructs a HabermasMachine with timing-wrapped clients."""
  if args.mock:
    raw_statement_client = types.LLMCLient.MOCK.get_client('mock')
    raw_reward_client = types.LLMCLient.MOCK.get_client('mock')
    statement_model = types.StatementModel.MOCK.get_model()
    reward_model = types.RewardModel.MOCK.get_model()
  else:
    raw_statement_client = types.LLMCLient.AISTUDIO.get_client(args.model)
    raw_reward_client = types.LLMCLient.AISTUDIO.get_client(args.model)
    statement_model = types.StatementModel.CHAIN_OF_THOUGHT.get_model()
    reward_model = types.RewardModel.CHAIN_OF_THOUGHT_RANKING.get_model()

  statement_client = TimingLLMClient(raw_statement_client)
  reward_client = TimingLLMClient(raw_reward_client)

  hm = machine.HabermasMachine(
      question=question,
      statement_client=statement_client,
      reward_client=reward_client,
      statement_model=statement_model,
      reward_model=reward_model,
      social_choice_method=types.RankAggregation.SCHULZE.get_method(
          tie_breaking_method=sc_utils.TieBreakingMethod.TIES_ALLOWED
      ),
      num_candidates=args.num_candidates,
      num_citizens=args.num_citizens,
      seed=0,
      max_workers=args.max_workers,
  )
  return hm, statement_client, reward_client


DEFAULT_QUESTION = 'Should public libraries stay open on Sundays?'
DEFAULT_OPINIONS = [
    'Yes, Sundays are when working people actually have time to visit.',
    'No, staffing costs are too high for the weekend demand we see.',
    'Only if the community demonstrates sustained demand first.',
    'Yes, libraries are one of the last free public spaces.',
    'Open a few branches on rotation rather than all of them.',
]
DEFAULT_CRITIQUES = [
    'This statement ignores the staffing cost concern.',
    'It should acknowledge that demand varies by branch.',
    'I would like a stronger nod to equity of access.',
    'The wording is fine but too vague on implementation.',
    'Consider mentioning a trial-period approach.',
]


def _pad(values: list[str], n: int) -> list[str]:
  return [values[i % len(values)] for i in range(n)]


def _load_from_sheet(
    url: str,
    label: str,
    opinion_column: str,
    question_column: str,
) -> tuple[str | None, list[str]]:
  """Fetches (question, values) from a public Google Sheet."""
  from sheets_io import fetch_from_google_sheets
  print(f'\nFetching {label} from Google Sheets...')
  start = time.perf_counter()
  question, values = fetch_from_google_sheets(
      url,
      opinion_column=opinion_column,
      question_column=question_column,
  )
  duration = time.perf_counter() - start
  print(f'  fetched {len(values)} row(s) in {duration:.2f}s')
  if question:
    print(f'  question from sheet: {question}')
  for i, v in enumerate(values, 1):
    preview = v[:100] + ('...' if len(v) > 100 else '')
    print(f'  [{i}] {preview}')
  return question, values


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--model', default='gemini-flash-latest',
                      help='Gemini model name for real API runs.')
  parser.add_argument('--num-citizens', type=int, default=3)
  parser.add_argument('--num-candidates', type=int, default=4)
  parser.add_argument('--mock', action='store_true',
                      help='Use mock clients/models (no API, no cost).')
  parser.add_argument('--skip-critique', action='store_true',
                      help='Only run the opinion round.')
  parser.add_argument('--api-key', default=None,
                      help='Google API key (else read from GOOGLE_API_KEY).')
  parser.add_argument('--opinions-sheet', default=None,
                      help='Public Google Sheets URL to pull opinions from. '
                           'Overrides the built-in defaults and sets '
                           'num_citizens to the number of rows fetched.')
  parser.add_argument('--critiques-sheet', default=None,
                      help='Public Google Sheets URL to pull critiques from.')
  parser.add_argument('--opinion-column', default='B',
                      help='Column letter for opinions/critiques (default B).')
  parser.add_argument('--question-column', default='A',
                      help="Column letter for question (default A, or '' to "
                           'skip).')
  parser.add_argument('--sheets-only', action='store_true',
                      help='Fetch from sheets and print what was parsed, '
                           'then exit without running the pipeline.')
  parser.add_argument('--max-workers', type=int, default=1,
                      help='Concurrent ranking LLM calls (default 1 = '
                           'serial). Capped at num_citizens.')
  args = parser.parse_args()

  needs_api = not args.mock and not args.sheets_only
  if needs_api:
    if args.api_key:
      os.environ['GOOGLE_API_KEY'] = args.api_key
    if not os.environ.get('GOOGLE_API_KEY'):
      print('GOOGLE_API_KEY not set. Re-run with --mock for a logic-only '
            'smoke test, or export GOOGLE_API_KEY to run against the real '
            'Gemini API.')
      return 0

  question: str | None = DEFAULT_QUESTION
  opinions: list[str] | None = None
  critiques: list[str] | None = None

  if args.opinions_sheet:
    sheet_question, opinions = _load_from_sheet(
        args.opinions_sheet, 'opinions',
        args.opinion_column, args.question_column,
    )
    if sheet_question:
      question = sheet_question
    if not opinions:
      print('ERROR: no opinions found in the sheet.')
      return 1

  if args.critiques_sheet:
    _, critiques = _load_from_sheet(
        args.critiques_sheet, 'critiques',
        args.opinion_column, args.question_column,
    )
    if not critiques:
      print('ERROR: no critiques found in the sheet.')
      return 1

  if args.sheets_only:
    print('\n--sheets-only: skipping pipeline.')
    return 0

  # If we pulled opinions from a sheet, let that define num_citizens so the
  # data lines up with the machine's expected count.
  if opinions is not None:
    args.num_citizens = len(opinions)
  else:
    opinions = _pad(DEFAULT_OPINIONS, args.num_citizens)

  if critiques is not None:
    if len(critiques) != args.num_citizens:
      print(f'ERROR: got {len(critiques)} critiques but num_citizens is '
            f'{args.num_citizens} (must match).')
      return 1
  else:
    critiques = _pad(DEFAULT_CRITIQUES, args.num_citizens)

  mode = 'mock' if args.mock else f'real API ({args.model})'
  print(f'\nSmoke test: {mode}, '
        f'{args.num_citizens} citizens, {args.num_candidates} candidates, '
        f'max_workers={args.max_workers}')
  print(f'Question: {question}')

  hm, statement_client, reward_client = build_machine(args, question)

  results = []

  opinion_result = run_round(
      hm, 'Opinion round', opinions,
      statement_client, reward_client,
      args.num_candidates, args.num_citizens,
  )
  results.append(opinion_result)
  print_summary(opinion_result)

  if not args.skip_critique:
    critique_result = run_round(
        hm, 'Critique round', critiques,
        statement_client, reward_client,
        args.num_candidates, args.num_citizens,
    )
    results.append(critique_result)
    print_summary(critique_result)

  grand_total = sum(r['total_s'] for r in results)
  print(f'\nGrand total: {grand_total:.2f}s across {len(results)} round(s)')

  all_ok = all(ok for r in results for _, ok in r['checks'])
  return 0 if all_ok else 1


if __name__ == '__main__':
  sys.exit(main())
