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

"""A ranking model that uses chain-of-thought reasoning."""

from collections.abc import Sequence
import re

import numpy as np
from typing_extensions import override

from habermas_machine.llm_client import base_client
from habermas_machine.reward_model import base_model


class COTRankingModel(base_model.BaseRankingModel):
  """A ranking model that uses chain-of-thought reasoning to rank statements."""

  @override
  def predict_ranking(
      self,
      llm_client: base_client.LLMClient,
      question: str,
      opinion: str,
      statements: Sequence[str],
      previous_winner: str | None = None,
      critique: str | None = None,
      seed: int | None = None,
      num_retries_on_error: int | None = None,
  ) -> base_model.RankingResult:
    """Ranks statements based on their length (see base class)."""
    if num_retries_on_error is None:
      num_retries_on_error = 0
    else:
      if num_retries_on_error < 0:
        raise ValueError('num_retries_on_error must be None or at least 0.')
    if previous_winner is None and critique is not None:
      raise ValueError(
          'If there is a previous_winner, there should be a critique.'
      )
    if len(statements) < 2:
      raise ValueError('There should be at least two statements to rank.')

    prompt = _generate_prompt(
        question, opinion, statements, previous_winner, critique
    )

    ranking_result = base_model.RankingResult(None, None)  # Dummy result.
    for _ in range(num_retries_on_error + 1):
      response = llm_client.sample_text(
          prompt, terminators=[], seed=seed, max_tokens=8192)
      ranking_result = _process_model_response(response, len(statements))

      if (
          ranking_result.ranking is not None
          and 'INCORRECT' not in ranking_result.explanation
      ):
        return ranking_result
      else:
        if seed is not None:
          seed += 1
          print(
              'Retrying with new seed. Explanation:'
              f' {ranking_result.explanation}'
          )

    # If we reach here, all retries failed. return the last result.
    return ranking_result


def _generate_opinion_critique_prompt(
    question: str,
    opinion: str,
    statements: Sequence[str],
    previous_winner: str,
    critique: str,
) -> str:
  """Generates a prompt for the LLM using opinion and critique."""
  n = len(statements)
  last_letter = chr(ord('A') + n - 1)
  full_alphabet = ', '.join(chr(ord('A') + i) for i in range(n))
  prompt = f"""As an AI assistant, your job is to rank these statements in the order that the participant would most likely agree with them, based on their opinion and critique to a summary statement from a previous discussion round. Use Arrow notation for the ranking, where ">" means "preferred to". Ties are NOT allowed and items should be in descending order of preference so you can ONLY use ">" and the letters of the statements in the ranking.

CRITICAL: Your ranking MUST include ALL {n} letters ({full_alphabet}) — every letter from A to {last_letter}, each appearing EXACTLY ONCE. A ranking that omits any letter is invalid and will be rejected. Even if some statements seem clearly worse than others, you MUST still place every letter somewhere in the ordering.

You MUST provide your answer in the following format. Output the Final Ranking FIRST, then the Reasoning:

## Final Ranking:
[ranking using arrow notation ONLY - all {n} letters separated by > symbols, nothing else]

## Reasoning:
[Brief explanation - one sentence per statement is sufficient]

For example, for five statements A, B, C, D and E:

## Final Ranking:
A > D > B > C > E

## Reasoning:
A aligns most closely with the participant's emphasis on immediate climate action and addresses the critique's call for concrete solutions. D and B acknowledge the issue but offer less concrete solutions. C downplays urgency. E opposes the participant's view entirely.

Question: {question}

Participant's Opinion: {opinion}

Statement from previous round: {previous_winner}

Critique: {critique}

Statements to rank:
"""
  for i, statement in enumerate(statements):
    letter = chr(ord('A') + i)  # A, B, C, D, etc.
    try:
      statement = (
          statement.strip().strip('').strip('""').strip('\n').strip()
      )
    except Exception as exc:
      raise ValueError(f'Issue with statement: {statement}') from exc
    prompt += f'{letter}. {statement}\n'

  return prompt.strip()


def _generate_opinion_only_prompt(
    question: str,
    opinion: str,
    statements: Sequence[str],
) -> str:
  """Generates a prompt for the LLM using only the opinion."""
  n = len(statements)
  last_letter = chr(ord('A') + n - 1)
  full_alphabet = ', '.join(chr(ord('A') + i) for i in range(n))
  prompt = f"""As an AI assistant, your job is to rank these statements in the order that the participant would most likely agree with them, based on their opinion. Use Arrow notation for the ranking, where ">" means "preferred to". Ties are NOT allowed and items should be in descending order of preference so you can ONLY use ">" and the letters of the statements in the ranking.

CRITICAL: Your ranking MUST include ALL {n} letters ({full_alphabet}) — every letter from A to {last_letter}, each appearing EXACTLY ONCE. A ranking that omits any letter is invalid and will be rejected. Even if some statements seem clearly worse than others, you MUST still place every letter somewhere in the ordering.

You MUST provide your answer in the following format. Output the Final Ranking FIRST, then the Reasoning:

## Final Ranking:
[ranking using arrow notation ONLY - all {n} letters separated by > symbols, nothing else]

## Reasoning:
[Brief explanation - one sentence per statement is sufficient]

For example, for five statements A, B, C, D and E:

## Final Ranking:
A > D > B > C > E

## Reasoning:
A aligns most closely with the participant's emphasis on immediate climate action. D and B acknowledge the issue but offer less concrete solutions. C downplays urgency. E opposes the participant's view entirely.

Question: {question}

Participant's Opinion: {opinion}

Statements to rank:
"""
  for i, statement in enumerate(statements):
    letter = chr(ord('A') + i)  # A, B, C, D, etc.
    try:
      statement = (
          statement.strip().strip('').strip('""').strip('\n').strip()
      )
    except Exception as exc:
      raise ValueError(f'Issue with statement: {statement}') from exc
    prompt += f'{letter}. {statement}\n'

  return prompt.strip()


def _generate_prompt(
    question: str,
    opinion: str,
    statements: Sequence[str],
    previous_winner: str | None = None,
    critique: str | None = None,
) -> str:
  """Generates a prompt for the LLM."""
  if previous_winner is None:
    return _generate_opinion_only_prompt(
        question, opinion, statements
    )
  else:
    return _generate_opinion_critique_prompt(
        question, opinion, statements, previous_winner, critique
    )


def _check_response_format(response: str) -> bool:
  """Checks if the response is in a correct format with <answer> and <sep>.

  Args:
    response: The model's raw response

  Returns:
    bool: True if the format is correct, False otherwise
  """
  pattern = r'<answer>\s*.*?\s*<sep>\s*.*?\s*</answer>'
  return bool(re.search(pattern, response, re.DOTALL))


def _check_arrow_format(arrow_ranking):
  """Checks if the arrow ranking format is correct.

  Args:
    arrow_ranking: The arrow ranking string (eg A > B > C)

  Returns:
  bool: True if the format is correct, False otherwise
  """
  if len(arrow_ranking) < 3:
    return False

  # Remove whitespace and replace multiple spaces with single spaces.
  arrow_ranking = re.sub(r'\s+', ' ', arrow_ranking.strip())

  # Remove spaces around '>' and '=' symbols.
  arrow_ranking = re.sub(r'\s*(>|=)\s*', r'\1', arrow_ranking)

  # Check if the ranking contains only allowed characters.
  if not re.match(r'^[A-Z>=]+$', arrow_ranking):
    return False

  # Check for consecutive '>' symbols, '=' at the start/end,
  # or '=' immediately before '>'.
  if (
      '>>' in arrow_ranking
      or arrow_ranking.startswith('=')
      or arrow_ranking.endswith('=')
      or '=>' in arrow_ranking
  ):
    return False

  # Split by '>' and check each group
  groups = arrow_ranking.split('>')

  if len(groups) < 1:
    return False

  seen_letters = set()
  for group in groups:
    # Check if the group is empty.
    if not group:
      return False
    # Check if each group contains only unique letters separated by '='.
    letters = group.split('=')
    if len(letters) != len(set(letters)):
      return False
    # Check if any letter in this group has been seen before.
    if any(letter in seen_letters for letter in letters):
      return False
    seen_letters.update(letters)

  return True


def _extract_arrow_ranking(text: str) -> str | None:
  """Extracts the arrow ranking from a given string.

  Args:
    text: The input string containing the arrow ranking.

  Returns:
    The extracted arrow ranking or None if not found.
  """
  # Regular expression to match a full arrow ranking pattern
  match = re.search(r'\b([A-Z](?:\s*(?:>|=)\s*[A-Z])*)\b', text)

  if match:
    return match.group(1).replace(' ', '')  # Removes any extra spaces
  else:
    return None


def _process_model_response(
    response: str, num_statements: int) -> base_model.RankingResult:
  """Processes the model's response, extract the explanation and arrow ranking.

  Args:
    response: The raw model response.
    num_statements: The number of statements to rank.

  Returns:
    A base_model.RankingResult of:
    - np.ndarray: The arrow ranking if it is correct, None otherwise.
    - str: The explanation if it is correct, "INCORRECT_TEMPLATE" if the
    response format is incorrect, or "INCORRECT_ARROW_RANKING" if the arrow
    ranking is incorrect.
  """
  # Try markdown format: Final Ranking first, then Reasoning
  ranking_first_match = re.search(r'##\s*Final Ranking:\s*(.*?)##\s*Reasoning:\s*(.*?)$', response, re.DOTALL | re.IGNORECASE)
  # Also try old order: Reasoning first, then Final Ranking
  reasoning_first_match = re.search(r'##\s*Reasoning:\s*(.*?)##\s*Final Ranking:\s*(.*?)(?:\n|$)', response, re.DOTALL | re.IGNORECASE)
  if ranking_first_match:
    arrow_ranking = _extract_arrow_ranking(ranking_first_match.group(1).strip())
    explanation = ranking_first_match.group(2).strip()
  elif reasoning_first_match:
    explanation = reasoning_first_match.group(1).strip()
    arrow_ranking = _extract_arrow_ranking(reasoning_first_match.group(2).strip())
  # Fall back to old XML-like format for backward compatibility
  elif _check_response_format(response):
    match = re.search(
        r'<answer>\s*(.*?)\s*<sep>\s*(.*?)\s*</answer>', response, re.DOTALL
    )
    if match is None:
      return base_model.RankingResult(None, f'INCORRECT_TEMPLATE: {response}')
    else:
      explanation = match.group(1).strip()
      arrow_ranking = _extract_arrow_ranking(
          match.group(2).strip())
  else:
    # Backup: look for "final ranking:" anywhere in response
    match = re.search(r'(?i)final ranking:\s*(.*)', response)
    if match is None:
      return base_model.RankingResult(None, f'INCORRECT_TEMPLATE: {response}')
    else:
      explanation = response
      arrow_ranking = _extract_arrow_ranking(match.group(1))

  if arrow_ranking is None or not _check_arrow_format(
      arrow_ranking
  ):
    # Check if the ranking is in the explanation.
    arrow_ranking = _extract_arrow_ranking(
        explanation.strip()
    )
    if arrow_ranking is None or not _check_arrow_format(
        arrow_ranking
    ):
      return base_model.RankingResult(
          None, f'INCORRECT_ARROW_RANKING: {response}')

  # Convert arrow ranking to numpy array.
  elements = re.findall(r'[A-Z]', arrow_ranking)
  unique_elements = sorted(set(elements))
  ranking_dict = {element: 0 for element in unique_elements}

  groups = arrow_ranking.split('>')
  for rank, group in enumerate(groups):
    tied_elements = group.strip().split('=')
    for element in tied_elements:
      ranking_dict[element.strip()] = rank

  result = np.array([ranking_dict[element] for element in unique_elements])

  if len(result) != num_statements:
    return base_model.RankingResult(
        None, f'INCORRECT_RANKING_LENGTH: {response}')

  return base_model.RankingResult(result, response)
