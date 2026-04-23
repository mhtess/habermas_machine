"""Compare Gemini-mediated Habermas Machine winners against the paper's winners.

For each example in habermas_machine_examples.md, this script:
  1. Runs HabermasMachine.mediate() on the participant opinions with a chosen
     Gemini model (same code path used by classroom_app.py).
  2. Asks the same model's reward model to rank the model's winner against the
     paper's reported winner from each citizen's perspective.
  3. Aggregates the per-citizen rankings via Schulze and reports the head-to-head
     verdict per example, plus a roll-up across examples.

Only the opinion round (initial winners) is compared.

Usage:
  python compare_statements.py --model gemini-flash-latest
  python compare_statements.py --model gemini-2.5-flash --example 1 --example 3
  python compare_statements.py --num-candidates 4 --max-workers 4 --json-out r.json
"""

import argparse
import concurrent.futures
import dataclasses
import json
import os
import re
import sys
import time
from typing import Any

import numpy as np

from habermas_machine import machine
from habermas_machine import types
from habermas_machine.social_choice import utils as sc_utils


EXAMPLES_PATH = 'habermas_machine_examples.md'

# A WINNER marker line looks like one of:
#   **★ SFT+RM (Habermas Machine) — WINNER:**
#   **★ SFT — WINNER:**
#   **★ SFT-RM — WINNER (initial round):**
# Followed by a blockquote of one or more `> ...` lines.
_WINNER_LINE = re.compile(r'^\*\*★[^*]*WINNER[^*]*:\*\*\s*$')
_OPINION_LINE = re.compile(r'^\*\*P(\d+):\*\*\s*(.*)$')


@dataclasses.dataclass
class Example:
  id: int
  title: str
  question: str
  opinions: list[str]
  paper_winner_initial: str


def _strip_blockquote(lines: list[str]) -> str:
  """Joins consecutive `> ...` lines into a single string."""
  out = []
  for ln in lines:
    if ln.startswith('>'):
      out.append(ln[1:].lstrip())
    elif ln.strip() == '':
      continue
    else:
      break
  return ' '.join(s for s in (l.strip() for l in out) if s)


def _section(body: str, heading: str) -> str | None:
  """Returns the text under `### heading` until the next `###` or end."""
  pattern = re.compile(
      rf'^###\s+{re.escape(heading)}\s*$(.*?)(?=^###\s|\Z)',
      re.MULTILINE | re.DOTALL,
  )
  m = pattern.search(body)
  return m.group(1) if m else None


def _parse_question(body: str) -> str:
  m = re.search(r'^\*\*Question:\*\*\s*(.+?)\s*$', body, re.MULTILINE)
  if not m:
    raise ValueError('Could not find **Question:** line.')
  return m.group(1).strip()


def _parse_opinions(body: str) -> list[str]:
  section = _section(body, 'Original opinions')
  if section is None:
    raise ValueError('Could not find ### Original opinions section.')
  opinions: dict[int, str] = {}
  current_p: int | None = None
  buf: list[str] = []
  for ln in section.splitlines():
    m = _OPINION_LINE.match(ln)
    if m:
      if current_p is not None:
        opinions[current_p] = ' '.join(buf).strip()
      current_p = int(m.group(1))
      buf = [m.group(2)]
    elif current_p is not None:
      buf.append(ln)
  if current_p is not None:
    opinions[current_p] = ' '.join(buf).strip()
  return [opinions[k] for k in sorted(opinions)]


def _parse_initial_winner(body: str) -> str:
  section = _section(body, 'Initial group statements')
  if section is None:
    raise ValueError('Could not find ### Initial group statements section.')
  lines = section.splitlines()
  for i, ln in enumerate(lines):
    if _WINNER_LINE.match(ln.strip()):
      text = _strip_blockquote(lines[i + 1:])
      if not text:
        raise ValueError('WINNER marker found but no blockquote followed.')
      return text
  raise ValueError('No WINNER marker found in initial statements.')


def parse_examples(path: str) -> list[Example]:
  with open(path, 'r', encoding='utf-8') as f:
    text = f.read()
  # Split on `## Example N: Title`. The first chunk is the file preamble.
  parts = re.split(r'^## Example (\d+):\s*(.+?)\s*$', text, flags=re.MULTILINE)
  out: list[Example] = []
  for i in range(1, len(parts), 3):
    eid = int(parts[i])
    title = parts[i + 1].strip()
    body = parts[i + 2]
    out.append(Example(
        id=eid,
        title=title,
        question=_parse_question(body),
        opinions=_parse_opinions(body),
        paper_winner_initial=_parse_initial_winner(body),
    ))
  return out


def _build_machine(
    model_name: str, question: str, num_candidates: int,
    num_citizens: int, max_workers: int, seed: int,
) -> machine.HabermasMachine:
  return machine.HabermasMachine(
      question=question,
      statement_client=types.LLMCLient.AISTUDIO.get_client(model_name),
      reward_client=types.LLMCLient.AISTUDIO.get_client(model_name),
      statement_model=types.StatementModel.CHAIN_OF_THOUGHT.get_model(),
      reward_model=types.RewardModel.CHAIN_OF_THOUGHT_RANKING.get_model(),
      social_choice_method=types.RankAggregation.SCHULZE.get_method(
          tie_breaking_method=sc_utils.TieBreakingMethod.TIES_ALLOWED,
      ),
      num_candidates=num_candidates,
      num_citizens=num_citizens,
      seed=seed,
      max_workers=max_workers,
  )


def _head_to_head_rankings(
    model_name: str,
    question: str,
    opinions: list[str],
    candidates: list[str],
    max_workers: int,
    seed: int,
) -> tuple[np.ndarray, list[int | None]]:
  """Returns (rankings[num_citizens, 2], errors-per-citizen).

  Order in `candidates` is canonical (index 0 = model, 1 = paper). The order
  shown to the LLM is shuffled per citizen to avoid position bias, then
  unshuffled before being returned.
  """
  rng = np.random.default_rng(seed)
  reward_model = types.RewardModel.CHAIN_OF_THOUGHT_RANKING.get_model()
  reward_client = types.LLMCLient.AISTUDIO.get_client(model_name)

  tasks = []
  for i, opinion in enumerate(opinions):
    perm = rng.permutation(len(candidates))
    shuffled = [candidates[j] for j in perm]
    tasks.append((i, opinion, perm, shuffled, int(rng.integers(2**31 - 1))))

  def run_one(task):
    i, opinion, perm, shuffled, s = task
    result = reward_model.predict_ranking(
        llm_client=reward_client,
        question=question,
        opinion=opinion,
        statements=shuffled,
        seed=s,
        num_retries_on_error=4,
    )
    return i, perm, result

  rankings = np.full((len(opinions), len(candidates)), sc_utils.RANKING_MOCK)
  errors: list[int | None] = [None] * len(opinions)
  workers = max(1, min(max_workers, len(opinions)))
  with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
    for i, perm, result in pool.map(run_one, tasks):
      if result.ranking is None:
        errors[i] = 1
        continue
      unshuffled = np.full_like(result.ranking, sc_utils.RANKING_MOCK)
      unshuffled[perm] = result.ranking
      rankings[i] = unshuffled
  return rankings, errors


def _schulze_winner(rankings: np.ndarray, seed: int) -> str:
  """Returns 'MODEL', 'PAPER', or 'TIE' for a 2-candidate head-to-head."""
  filtered = sc_utils.filter_out_mocks(rankings)
  if filtered.shape[0] == 0:
    return 'TIE'
  schulze = types.RankAggregation.SCHULZE.get_method(
      tie_breaking_method=sc_utils.TieBreakingMethod.TIES_ALLOWED,
  )
  tied, _untied = schulze.aggregate(filtered, seed=seed)
  if tied[0] < tied[1]:
    return 'MODEL'
  if tied[1] < tied[0]:
    return 'PAPER'
  return 'TIE'


def _first_place_tally(rankings: np.ndarray) -> dict[str, int]:
  tally = {'model': 0, 'paper': 0, 'tied': 0, 'errored': 0}
  for row in rankings:
    if int(row[0]) == sc_utils.RANKING_MOCK:
      tally['errored'] += 1
    elif row[0] < row[1]:
      tally['model'] += 1
    elif row[1] < row[0]:
      tally['paper'] += 1
    else:
      tally['tied'] += 1
  return tally


def compare_one(
    example: Example, model_name: str, num_candidates: int,
    max_workers: int, seed: int,
) -> dict[str, Any]:
  start = time.perf_counter()
  hm = _build_machine(
      model_name=model_name,
      question=example.question,
      num_candidates=num_candidates,
      num_citizens=len(example.opinions),
      max_workers=max_workers,
      seed=seed,
  )
  model_winner, _ = hm.mediate(example.opinions)
  mediate_s = time.perf_counter() - start

  rank_start = time.perf_counter()
  rankings, errors = _head_to_head_rankings(
      model_name=model_name,
      question=example.question,
      opinions=example.opinions,
      candidates=[model_winner, example.paper_winner_initial],
      max_workers=max_workers,
      seed=seed + 1,
  )
  rank_s = time.perf_counter() - rank_start

  tally = _first_place_tally(rankings)
  verdict = _schulze_winner(rankings, seed=seed + 2)
  return {
      'example_id': example.id,
      'title': example.title,
      'question': example.question,
      'num_citizens': len(example.opinions),
      'model_winner': model_winner,
      'paper_winner': example.paper_winner_initial,
      'rankings': rankings.tolist(),
      'first_place': tally,
      'verdict': verdict,
      'mediate_s': mediate_s,
      'rank_s': rank_s,
      'errors_per_citizen': sum(1 for e in errors if e),
  }


def _preview(s: str, n: int = 200) -> str:
  s = s.strip().replace('\n', ' ')
  return s if len(s) <= n else s[:n - 1] + '…'


def print_result(r: dict[str, Any]) -> None:
  print(f'\n=== Example {r["example_id"]}: {r["title"]} ===')
  print(f'  question: {r["question"]}')
  print(f'  citizens: {r["num_citizens"]}'
        f'  (mediate {r["mediate_s"]:.1f}s, rank {r["rank_s"]:.1f}s)')
  print(f'  Model winner:  "{_preview(r["model_winner"])}"')
  print(f'  Paper winner:  "{_preview(r["paper_winner"])}"')
  fp = r['first_place']
  print(f'  Per-citizen first-place: model={fp["model"]} paper={fp["paper"]} '
        f'tied={fp["tied"]} errored={fp["errored"]}')
  print(f'  Schulze head-to-head: {r["verdict"]}')


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--model', default='gemini-flash-latest',
                      help='Gemini model name.')
  parser.add_argument('--example', type=int, action='append', default=None,
                      help='Probe only this example id (repeatable).')
  parser.add_argument('--num-candidates', type=int, default=4,
                      help='Candidates per mediation round (default 4).')
  parser.add_argument('--max-workers', type=int, default=4,
                      help='Concurrent ranking LLM calls (default 4).')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--api-key', default=None)
  parser.add_argument('--md-path', default=EXAMPLES_PATH)
  parser.add_argument('--json-out', default=None,
                      help='Optional path to dump full results as JSON.')
  args = parser.parse_args()

  if args.api_key:
    os.environ['GOOGLE_API_KEY'] = args.api_key
  if not os.environ.get('GOOGLE_API_KEY'):
    print('GOOGLE_API_KEY not set. Export it or pass --api-key.')
    return 2

  examples = parse_examples(args.md_path)
  if args.example:
    wanted = set(args.example)
    examples = [e for e in examples if e.id in wanted]
    missing = wanted - {e.id for e in examples}
    if missing:
      print(f'Unknown example id(s): {sorted(missing)}')
      return 2
  if not examples:
    print('No examples to compare.')
    return 2

  print(f'Comparing {len(examples)} example(s) using model: {args.model}')
  print(f'  num_candidates={args.num_candidates}  '
        f'max_workers={args.max_workers}  seed={args.seed}')

  results = []
  for i, ex in enumerate(examples):
    try:
      r = compare_one(
          ex, args.model, args.num_candidates, args.max_workers,
          seed=args.seed + i * 100,
      )
    except Exception as e:  # pylint: disable=broad-except
      print(f'\n=== Example {ex.id}: {ex.title} === FAILED: '
            f'{type(e).__name__}: {e}')
      results.append({'example_id': ex.id, 'title': ex.title,
                      'error': f'{type(e).__name__}: {e}'})
      continue
    print_result(r)
    results.append(r)

  scoreboard = {'MODEL': 0, 'PAPER': 0, 'TIE': 0, 'FAILED': 0}
  for r in results:
    if 'error' in r:
      scoreboard['FAILED'] += 1
    else:
      scoreboard[r['verdict']] += 1
  print(f'\n=== Summary across {len(results)} example(s) ===')
  print(f'  Model wins: {scoreboard["MODEL"]}')
  print(f'  Paper wins: {scoreboard["PAPER"]}')
  print(f'  Ties:       {scoreboard["TIE"]}')
  if scoreboard['FAILED']:
    print(f'  Failed:     {scoreboard["FAILED"]}')

  if args.json_out:
    with open(args.json_out, 'w', encoding='utf-8') as f:
      json.dump({'model': args.model, 'results': results,
                 'scoreboard': scoreboard}, f, indent=2)
    print(f'\nWrote full results to {args.json_out}')

  return 0 if scoreboard['FAILED'] == 0 else 1


if __name__ == '__main__':
  sys.exit(main())
