"""Probe each Gemini model listed in the classroom app to confirm it works.

For every model in MODELS, this sends a trivial prompt through AIStudioClient
and reports whether it returned non-empty text. Requires GOOGLE_API_KEY.

Usage:
  python test_gemini_models.py
  python test_gemini_models.py --model gemini-2.5-pro
  python test_gemini_models.py --api-key ...

Exit code is 0 iff every probed model succeeded.
"""

import argparse
import os
import sys
import time
import traceback

from habermas_machine.llm_client import aistudio_client


# Keep in sync with classroom_app.py's model selectbox.
MODELS = (
    'gemini-2.5-flash-lite',
    'gemini-flash-latest',
    'gemini-pro-latest',
    'gemini-3-flash-preview',
    'gemini-3-pro-preview',
    'gemini-2.5-flash',
    'gemini-2.5-pro',
)

PROBE_PROMPT = 'Reply with the single word: ok'


def probe(model_name: str, max_tokens: int, temperature: float) -> dict:
  """Runs one probe call and returns a result dict."""
  result = {
      'model': model_name,
      'ok': False,
      'duration_s': 0.0,
      'response': '',
      'error': None,
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
    # AIStudioClient swallows ValueError and returns '' — treat that as a fail.
    result['ok'] = bool(response.strip())
    if not result['ok']:
      result['error'] = 'empty response'
  except Exception as e:  # pylint: disable=broad-except
    result['error'] = f'{type(e).__name__}: {e}'
    result['traceback'] = traceback.format_exc()
  result['duration_s'] = time.perf_counter() - start
  return result


def print_row(r: dict) -> None:
  mark = 'PASS' if r['ok'] else 'FAIL'
  line = f'  [{mark}] {r["model"]:<28} {r["duration_s"]:5.2f}s'
  if r['ok']:
    preview = r['response'].strip().replace('\n', ' ')[:60]
    line += f'  -> {preview!r}'
  else:
    line += f'  -> {r["error"]}'
  print(line)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--model', action='append', default=None,
      help='Probe only this model (repeatable). Default: all models.',
  )
  parser.add_argument('--api-key', default=None,
                      help='Google API key (else read from GOOGLE_API_KEY).')
  parser.add_argument('--max-tokens', type=int, default=16)
  parser.add_argument('--temperature', type=float, default=0.0)
  parser.add_argument('--verbose', action='store_true',
                      help='Print full traceback on failure.')
  args = parser.parse_args()

  if args.api_key:
    os.environ['GOOGLE_API_KEY'] = args.api_key
  if not os.environ.get('GOOGLE_API_KEY'):
    print('GOOGLE_API_KEY not set. Export it or pass --api-key.')
    return 2

  models = tuple(args.model) if args.model else MODELS
  print(f'Probing {len(models)} model(s) with prompt: {PROBE_PROMPT!r}\n')

  results = [probe(m, args.max_tokens, args.temperature) for m in models]
  for r in results:
    print_row(r)
    if args.verbose and not r['ok'] and 'traceback' in r:
      print(r['traceback'])

  passed = sum(1 for r in results if r['ok'])
  failed = len(results) - passed
  print(f'\n{passed} passed, {failed} failed')
  return 0 if failed == 0 else 1


if __name__ == '__main__':
  sys.exit(main())
