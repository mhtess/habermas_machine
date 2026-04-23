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

"""A model that uses the chain-of-thought method to generate statements."""

from collections.abc import Sequence

import re

from habermas_machine.llm_client import base_client
from habermas_machine.statement_model import base_model


def _generate_opinion_critique_prompt(
    question: str,
    opinions: Sequence[str],
    previous_winner: str,
    critiques: Sequence[str],
) -> str:
  """Generates a prompt using opinions, previous winner, and critiques."""

  prompt = f"""You are assisting a group of citizens in a deliberation process. A first draft of a group consensus statement was created, and the participants have now provided critiques of that draft. Your role is to generate a revised consensus statement that incorporates the critiques while preserving what worked well in the original.

IMPORTANT GUIDELINES FOR THE REVISED STATEMENT:
- Make targeted revisions to address the critiques. Do NOT rewrite the entire statement from scratch. Keep the parts that worked well and add or modify specific parts based on the critiques.
- Write in the first-person plural as the voice of the group: "We believe...", "We feel...", "We have come to the conclusion that..."
- The statement should read as a natural group opinion, not an academic analysis.
- Do NOT include any references to opinion numbers, participant numbers, or critique numbers in the final statement. These references belong only in the reasoning section.
- Acknowledge the majority view while also incorporating minority perspectives where possible.
- Be substantive and specific — include concrete positions or proposals, not vague hedging like "further investigation is needed."
- Aim for 100-200 words — a substantial paragraph that captures the group's nuanced position.
- If a critique is sarcastic, off-topic, or not constructive, do not incorporate it into the revision.

Please think through this task step-by-step:

1. Carefully review the previous draft consensus statement and identify its strengths.
2. Analyze the critiques, noting specific suggestions, concerns, and areas of agreement or disagreement with the draft.
3. Determine which critiques represent substantive suggestions that should be incorporated, and which (if any) are not constructive.
4. Generate a revised statement that keeps the strong parts of the original draft and makes targeted additions or modifications to address the substantive critiques.

Provide your answer in the following format:

## Reasoning:
[Your step-by-step reasoning, referencing specific opinion and critique numbers]

## Revised Consensus Statement:
[The revised consensus statement ONLY — written as the group's voice, with NO references to opinion or critique numbers]

Example:

## Reasoning:
1. The previous draft effectively captures the group's opposition to raising the retirement age and acknowledges the cost implications.
2. Critique 1 agrees with the statement and finds it succinct. Critique 2 suggests adding support for pensioners on state pension only. Critique 3 simply restates opposition. Critique 4 questions the relevance of the food legislation point.
3. Critiques 2's suggestion about pensioner support is substantive and should be added. Critique 4's concern about the food legislation point is valid but was supported in the original opinions, so it should remain but could be softened.
4. The revised statement keeps the original largely intact and adds one sentence addressing Critique 2's suggestion about pensioner support.

## Revised Consensus Statement:
In general, the group was opposed to raising the retirement age from 66 to 68 years. However, we recognised that people are living longer and this has implications for the cost of the state pension. We also recognised that many people are in poor health in later life and this can be due to poverty and lifestyle. We believe that the government should support people to stay healthy in later life through health advice and legislation on food producers. We believe that people should have the choice to retire at 66 or to continue to work if they wish to do so. We also believe that the government should consider more support for pensioners who have only the state pension to live on.

It is CRITICAL to follow this format. Always include the "## Reasoning:" section followed by your explanation, then the "## Revised Consensus Statement:" section with ONLY the statement. The final statement must NOT contain any references like "(Opinion 1)" or "(Critique 2)".


Below you will find the question, the individual opinions, the previous draft consensus statement, and the critiques provided by the participants.


Question: {question}

Individual Opinions:
"""
  for i, opinion in enumerate(opinions):
    prompt += f'Opinion Person {i+1}: {opinion}\n'

  prompt += f"""
Previous Draft Consensus Statement: {previous_winner}

Critiques of the Previous Draft:
"""

  for i, critique in enumerate(critiques):
    prompt += f'Critique Person {i+1}: {critique}\n'

  return prompt.strip()


def _generate_opinion_only_prompt(
    question: str,
    opinions: Sequence[str],
) -> str:
  """Generates a prompt for the LLM using only the opinions."""
  prompt = f"""You are assisting a group of citizens in a deliberation process. The participants have provided their individual opinions on an important question. Your role is to generate a draft consensus statement that captures the main points of agreement and represents the collective view of the group.

IMPORTANT GUIDELINES FOR THE CONSENSUS STATEMENT:
- Write in the first-person plural as the voice of the group: "We believe...", "We feel...", "In general, the group..."
- The statement should read as a natural group opinion, not an academic analysis.
- Do NOT include any references to opinion numbers or participant numbers in the final statement. These references belong only in the reasoning section.
- Acknowledge the majority view while also incorporating minority perspectives where possible. Use phrases like "However, we recognised that..." or "We also acknowledged that..." to show nuance.
- Be substantive and specific — include concrete positions or proposals, not vague hedging like "further investigation is needed."
- Aim for 100-200 words — a substantial paragraph that captures the group's nuanced position.
- The statement must not directly contradict any individual opinion.

Please think through this task step-by-step:

1. Carefully analyze the individual opinions, noting key themes, points of agreement, and areas of disagreement.
2. Identify the majority position and any significant minority perspectives that should be acknowledged.
3. Synthesize a consensus statement that represents the shared perspective while acknowledging where views differ.

Provide your answer in the following format:

## Reasoning:
[Your step-by-step reasoning, referencing specific opinion numbers]

## Draft Consensus Statement:
[The draft consensus statement ONLY — written as the group's voice, with NO references to opinion numbers]

Example:

## Reasoning:
1. Opinions 1, 2, and 4 support raising the retirement age, citing longer life expectancy and pension costs (majority view). Opinion 3 is neutral, acknowledging both sides.
2. Opinion 2 uniquely emphasizes the need for support for those in poor health who cannot work until 68 — this represents an important caveat.
3. The consensus should reflect the majority support for raising the age while acknowledging the health concerns raised in Opinion 2.

## Draft Consensus Statement:
We should raise the retirement age from 66 to 68. People are living longer, and we should expect them to work longer. This will also reduce the burden on the state to fund pensions. However, we recognised that many people are in poor health in later life and this can be due to poverty and lifestyle. We believe that the government should support people to stay healthy in later life and provide assistance for those unable to work until 68.

It is CRITICAL to follow this format. Always include the "## Reasoning:" section followed by your explanation, then the "## Draft Consensus Statement:" section with ONLY the statement. The final statement must NOT contain any references like "(Opinion 1)" or "(Opinions 2, 3)".


Below you will find the question and the individual opinions of the participants.

Question: {question}

Individual Opinions:
"""

  for i, opinion in enumerate(opinions):
    prompt += f'Opinion Person {i+1}: {opinion}\n'

  return prompt.strip()


def _generate_prompt(
    question: str,
    opinions: Sequence[str],
    previous_winner: str | None = None,
    critiques: Sequence[str] | None = None,
) -> str:
  """Generates a prompt for the LLM."""
  if previous_winner is None:
    return _generate_opinion_only_prompt(question, opinions)
  else:
    return _generate_opinion_critique_prompt(
        question, opinions, previous_winner, critiques
    )


def _process_model_response(response: str) -> tuple[str, str]:
  """Processes the model's response, extracting the statement and explanation.

  Args:
      response: The raw model response.

  Returns:
      A tuple of (statement, explanation).  If the response format is
      incorrect, returns ("", "INCORRECT_TEMPLATE").
  """
  # Try new markdown format first (either "Draft" or "Revised" Consensus Statement)
  match = re.search(
      r'##\s*Reasoning:\s*(.*?)##\s*(?:Draft|Revised)\s+Consensus Statement:\s*(.*?)(?:\n##|$)',
      response,
      re.DOTALL | re.IGNORECASE
  )
  if match:
    explanation = match.group(1).strip()
    statement = match.group(2).strip()
    return statement, explanation

  # Fall back to old XML-like format for backward compatibility
  match = re.search(
      r'<answer>\s*(.*?)\s*<sep>\s*(.*?)\s*</answer>', response, re.DOTALL
  )
  if match:
    explanation = match.group(1).strip()
    statement = match.group(2).strip()
    return statement, explanation

  return '', 'INCORRECT_TEMPLATE'


class COTModel(base_model.BaseStatementModel):
  """Statement model that uses chain-of-thought reasoning."""

  def generate_statement(
      self,
      llm_client: base_client.LLMClient,
      question: str,
      opinions: Sequence[str],
      previous_winner: str | None = None,
      critiques: Sequence[str] | None = None,
      seed: int | None = None,
      override_prompt: str | None = None,
      num_retries_on_error: int = 1,
  ) -> base_model.StatementResult:
    """Generates a statement (see base model)."""
    if num_retries_on_error is None:
      num_retries_on_error = 0
    else:
      if num_retries_on_error < 0:
        raise ValueError('num_retries_on_error must be None or at least 0.')
    prompt = _generate_prompt(question, opinions, previous_winner, critiques)
    statement, explanation = '', ''  # Dummy result.
    for _ in range(num_retries_on_error):
      response = llm_client.sample_text(
          prompt, terminators=[], seed=seed)

      statement, explanation = _process_model_response(response)
      if len(statement) > 5 and 'INCORRECT' not in explanation:
        return base_model.StatementResult(statement, explanation)
      else:
        if seed is not None:
          seed += 1
          print(f'Retrying with new seed. Explanation: {explanation}')

    # If we reach here, all retries failed. Return the last result.
    return base_model.StatementResult(statement, explanation)


