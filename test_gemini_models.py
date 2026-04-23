"""Probe each Gemini model listed in the classroom app to confirm it works.

For every model in aistudio_client.SUPPORTED_MODELS, this sends a trivial
prompt through AIStudioClient and reports PASS / FAIL / QUOTA. Requires
GOOGLE_API_KEY.

Usage:
  python test_gemini_models.py
  python test_gemini_models.py --model gemini-2.5-pro
  python test_gemini_models.py --verbose

Exit code is 0 iff every probed model succeeded (quota counts as failure).
"""

import argparse
import os
import sys
import time
import traceback

from habermas_machine.llm_client import aistudio_client


PROBE_PROMPT = 'Reply with the single word: ok'

# Status codes used in the report.
PASS = 'PASS'
FAIL = 'FAIL'
QUOTA = 'QUOTA'


def _classify(exc: BaseException) -> str:
  # google.api_core.exceptions.ResourceExhausted is the 429 quota error.
  if type(exc).__name__ == 'ResourceExhausted':
    return QUOTA
  return FAIL


def probe(model_name: str, max_tokens: int, temperature: float) -> dict:
  """Runs one probe call and returns a result dict."""
  result = {
      'model': model_name,
      'status': FAIL,
      'duration_s': 0.0,
      'response': '',
      'error': None,
      'traceback': None,
  }
  start = time.perf_counter()
  try:
    client = aistudio_client.AIStudioClient(model_name=model_name)
    response = client.sample_text(
        PROBE_PROMPT,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    result['response'] = response
    if response.strip():
      result['status'] = PASS
    else:
      # AIStudioClient swallows missing-text errors and returns ''.
      result['error'] = 'empty response'
  except Exception as e:  # pylint: disable=broad-except
    result['status'] = _classify(e)
    result['error'] = f'{type(e).__name__}: {e}'
    result['traceback'] = traceback.format_exc()
  result['duration_s'] = time.perf_counter() - start
  return result


def _short(msg: str, limit: int = 140) -> str:
  """Collapse multi-line errors to a single trimmed line."""
  first = msg.strip().splitlines()[0] if msg else ''
  return first if len(first) <= limit else first[:limit - 1] + '…'


def print_row(r: dict, verbose: bool) -> None:
  line = f'  [{r["status"]:<5}] {r["model"]:<28} {r["duration_s"]:5.2f}s'
  if r['status'] == PASS:
    preview = r['response'].strip().replace('\n', ' ')[:60]
    line += f'  -> {preview!r}'
  else:
    err = r['error'] or ''
    line += f'  -> {err if verbose else _short(err)}'
  print(line)
  if verbose and r['traceback']:
    print(r['traceback'])


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--model', action='append', default=None,
      help='Probe only this model (repeatable). Default: all models.',
  )
  parser.add_argument('--api-key', default=None,
                      help='Google API key (else read from GOOGLE_API_KEY).')
  parser.add_argument('--max-tokens', type=int, default=512,
                      help='Output token budget. Thinking models need '
                           'headroom or they emit no text. Default: 512.')
  parser.add_argument('--temperature', type=float, default=0.0)
  parser.add_argument('--verbose', action='store_true',
                      help='Print full error message and traceback on failure.')
  args = parser.parse_args()

  if args.api_key:
    os.environ['GOOGLE_API_KEY'] = args.api_key
  if not os.environ.get('GOOGLE_API_KEY'):
    print('GOOGLE_API_KEY not set. Export it or pass --api-key.')
    return 2

  models = tuple(args.model) if args.model else aistudio_client.SUPPORTED_MODELS
  print(f'Probing {len(models)} model(s) with prompt: {PROBE_PROMPT!r}\n')

  results = [probe(m, args.max_tokens, args.temperature) for m in models]
  for r in results:
    print_row(r, args.verbose)

  passed = sum(1 for r in results if r['status'] == PASS)
  quota = sum(1 for r in results if r['status'] == QUOTA)
  failed = sum(1 for r in results if r['status'] == FAIL)
  print(f'\n{passed} passed, {failed} failed, {quota} quota-exceeded')
  return 0 if (failed + quota) == 0 else 1


if __name__ == '__main__':
  sys.exit(main())
