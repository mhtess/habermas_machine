# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Language Model that uses GDM AI Studio API."""

from collections.abc import Collection, Mapping, Sequence
import os
import random
import time

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from typing_extensions import override

from habermas_machine.llm_client import base_client
from habermas_machine.llm_client import utils


# Gemini models exposed in the classroom UI and covered by the probe script.
SUPPORTED_MODELS = (
    'gemini-2.5-flash-lite',
    'gemini-flash-latest',
    'gemini-pro-latest',
    'gemini-3-flash-preview',
    'gemini-3-pro-preview',
    'gemini-2.5-flash',
    'gemini-2.5-pro',
)


# Errors that are worth retrying with backoff (transient: rate limits, server
# overload, deadline exceeded, network blips). Anything else is surfaced
# immediately so genuine bugs aren't masked by long retry loops.
_RETRYABLE_EXCEPTIONS = (
    google_exceptions.ResourceExhausted,    # 429 quota / rate limit
    google_exceptions.ServiceUnavailable,   # 503
    google_exceptions.DeadlineExceeded,     # 504
    google_exceptions.InternalServerError,  # 500
    google_exceptions.Aborted,              # 409, transient concurrency
)
_DEFAULT_MAX_RETRIES = 6
_DEFAULT_INITIAL_BACKOFF_S = 2.0
_DEFAULT_MAX_BACKOFF_S = 60.0


DEFAULT_SAFETY_SETTINGS = (
    {
        'category': 'HARM_CATEGORY_HARASSMENT',
        'threshold': 'BLOCK_ONLY_HIGH',
    },
    {
        'category': 'HARM_CATEGORY_HATE_SPEECH',
        'threshold': 'BLOCK_ONLY_HIGH',
    },
    {
        'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        'threshold': 'BLOCK_ONLY_HIGH',
    },
    {
        'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
        'threshold': 'BLOCK_ONLY_HIGH',
    },
)


class AIStudioClient(base_client.LLMClient):
  """Language Model that uses AI Studio API."""

  def __init__(
      self,
      model_name: str,
      *,
      safety_settings: Sequence[Mapping[str, str]] = DEFAULT_SAFETY_SETTINGS,
      sleep_periodically: bool = False,
      max_retries: int = _DEFAULT_MAX_RETRIES,
      initial_backoff_s: float = _DEFAULT_INITIAL_BACKOFF_S,
      max_backoff_s: float = _DEFAULT_MAX_BACKOFF_S,
  ) -> None:
    """Initializes the instance.

    Args:
      model_name: which language model to use. For more details, see
        https://aistudio.google.com/.
      safety_settings: Gemini safety settings. For more details, see
        https://ai.google.dev/gemini-api/docs/safety.
      sleep_periodically: sleep between API calls to avoid rate limit.
      max_retries: max retry attempts for transient errors (rate limits, 5xx).
      initial_backoff_s: initial backoff between retries; doubles each attempt.
      max_backoff_s: cap on per-attempt backoff.
    """
    self._api_key = os.environ['GOOGLE_API_KEY']
    self._model_name = model_name
    self._safety_settings = safety_settings
    self._sleep_periodically = sleep_periodically
    self._max_retries = max_retries
    self._initial_backoff_s = initial_backoff_s
    self._max_backoff_s = max_backoff_s

    genai.configure(api_key=self._api_key)
    self._model = genai.GenerativeModel(
        model_name=self._model_name,
        safety_settings=safety_settings,
    )

    self._calls_between_sleeping = 10
    self._n_calls = 0

  @override
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
    del timeout
    del seed  # AI Studio does not support seeds.

    self._n_calls += 1
    if self._sleep_periodically and (
        self._n_calls % self._calls_between_sleeping == 0):
      print('Sleeping for 10 seconds...')
      time.sleep(10)

    # Retry transient errors (rate limits, 5xx) with exponential backoff +
    # jitter. At scale (hundreds of concurrent ranking calls) 429s are routine
    # rather than exceptional, so the client must absorb them.
    sample = None
    backoff = self._initial_backoff_s
    last_exc: Exception | None = None
    for attempt in range(self._max_retries + 1):
      try:
        sample = self._model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                stop_sequences=terminators,
            ),
            safety_settings=self._safety_settings,
            stream=False,
        )
        break
      except _RETRYABLE_EXCEPTIONS as e:
        last_exc = e
        if attempt == self._max_retries:
          raise
        # Full jitter on top of exponential backoff.
        sleep_s = min(self._max_backoff_s, backoff) * (0.5 + random.random())
        print(
            f'Transient API error ({type(e).__name__}); retrying in '
            f'{sleep_s:.1f}s (attempt {attempt + 1}/{self._max_retries}).'
        )
        time.sleep(sleep_s)
        backoff = min(self._max_backoff_s, backoff * 2)
    if sample is None:
      # Should be unreachable: the loop either breaks or re-raises above.
      raise RuntimeError('LLM call failed without a response.') from last_exc
    try:
      # AI Studio returns a list of parts, but we only use the first one.
      response = sample.candidates[0].content.parts[0].text
    except (ValueError, IndexError, AttributeError) as e:
      # Empty `parts` happens e.g. when a thinking model exhausts
      # max_output_tokens on internal thoughts, or when safety filtering
      # strips the content. Surface the finish_reason so the cause is clear.
      finish_reason = None
      try:
        finish_reason = sample.candidates[0].finish_reason
      except (IndexError, AttributeError):
        pass
      print(f'No text in response (finish_reason={finish_reason}): {e}')
      print(f'prompt: {prompt}')
      print(f'sample: {sample}')
      response = ''
    return utils.truncate(response, delimiters=terminators)

