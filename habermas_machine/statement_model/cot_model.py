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

  prompt = f"""You are assisting a group of citizens. A first draft consensus statement was created, and participants have now critiqued it. Your job is to revise the statement so that more people will rank it first.

WHAT MAKES A WINNING REVISION:

- Make targeted edits, not a full rewrite. Keep the parts of the draft that worked and modify only the parts that need it.
- Borrow specifics from critiques. Where participants propose concrete things (numbers, mechanisms, examples, carve-outs), weave their actual words in. Do not paraphrase into generic language.
- Ignore sarcastic, off-topic, or non-constructive critiques. Not every critique deserves a response.
- Keep the stance clear. Revisions should sharpen the position, not hedge it into "both sides have a point". Minority concerns should be compressed into brief "however, we recognise..." clauses, not expanded into symmetric alternatives.
- Propose, don't just oppose. If the critique or the group rejects something, pair the revision with a positive alternative.
- Be concrete, not abstract. Favor specific mechanisms over vague principles.
- Length follows substance. Don't pad to incorporate every critique; don't delete specifics to shorten.
- The goal is first-place votes, not universal acceptance. Write to persuade.
- Write as the group in first-person plural: "We believe...", "We feel...", "We have come to the conclusion that...".
- Do NOT reference opinion, participant, or critique numbers in the final statement. Those belong only in the reasoning section.

Please think through this task step-by-step:

1. Identify the strengths of the previous draft — what should be preserved.
2. Categorize each critique as substantive (carries a concrete suggestion) or non-constructive (sarcastic, off-topic, restating).
3. For each substantive critique, identify the specific words or proposals to borrow.
4. Make targeted edits that address substantive critiques while keeping the draft's stance clear.

Provide your answer in the following format:

## Reasoning:
[Your step-by-step reasoning, referencing specific opinion and critique numbers]

## Revised Consensus Statement:
[The revised consensus statement ONLY — written as the group's voice, with NO references to opinion or critique numbers]

Example:

Question: Should the government provide universal free childcare from birth?

Individual Opinions:
Opinion Person 1: Free childcare is important but not from birth — babies benefit from a consistent primary caregiver. Offer universal paid parental leave from birth and universal free childcare from 6 months.
Opinion Person 2: Free childcare should be targeted at parents who can't afford it, not universal, and only from age three.
Opinion Person 3: Childcare costs are disproportionate to what one parent can earn. Drop-off and pick-up times limit careers.
Opinion Person 4: Parents often struggle between maternity leave ending and free childcare starting. Offer a choice of free childcare or a basic income.
Opinion Person 5: Childcare support helps parents and communities. Extended family isn't always available in the UK.

Previous Draft Consensus Statement: In general, free childcare is a good thing, but it is important to consider how it is provided and for which age groups. We feel that it is important to offer support to parents in the form of parental leave, and that this should be available to both parents. We also feel that free childcare should be provided in a way that supports children's development and learning, not just as a childminding service. However, we do not feel that free childcare should be provided from birth, as it is important for babies to have a consistent primary caregiver in their early months. For this reason, we would support the government providing universal paid parental leave from birth, and universal free childcare from 6 months old.

Critiques of the Previous Draft:
Critique Person 1: I agree. I would add that we support parents being able to access paid childcare from birth if they need it — just not universal free childcare from birth.
Critique Person 2: Good idea on paternity leave, but if the costs are not covered by government it could be crippling for small businesses.
Critique Person 3: Can we add something about being irrespective of gender?
Critique Person 4: We need to consider what form the childcare takes and how many hours would be free — it has to be enough to make going back to work worthwhile.
Critique Person 5: Agree. The option to exchange childcare for parental leave is a great idea.

## Reasoning:
1. The draft takes a clear stance (support childcare, not from birth) and already borrows "6 months" and "universal paid parental leave" — these are strengths to preserve.
2. Critique 1 agrees; the suggestion about access to paid childcare from birth "if needed" is a useful nuance. Critique 3 asks for an explicit "irrespective of gender" clause — concrete and easy to lift verbatim. Critique 5 endorses. Critique 2's small-business concern is off the central question. Critique 4 asks for more mechanism detail.
3. Specific words to borrow: "irrespective of gender" (Critique 3), "if they need it" framing (Critique 1).
4. Targeted edits: add a final sentence lifting "irrespective of gender" from Critique 3. Leave the draft's stance and structure intact.

## Revised Consensus Statement:
In general, free childcare is a good thing, but it is important to consider how it is provided and for which age groups. We feel that it is important to offer support to parents in the form of parental leave, and that this should be available to both parents. We also feel that free childcare should be provided in a way that supports children's development and learning, not just as a childminding service. However, we do not feel that free childcare should be provided from birth, as it is important for babies to have a consistent primary caregiver in their early months. For this reason, we would support the government providing universal paid parental leave from birth, and universal free childcare from 6 months old. In addition, we would like to stress that childcare and parental leave should be available to all parents, irrespective of gender.

It is CRITICAL to follow this format. Always include the "## Reasoning:" section followed by your explanation, then the "## Revised Consensus Statement:" section with ONLY the statement. The final statement must NOT contain any references like "(Opinion 1)" or "(Critique 2)".

================================================================================
END OF FORMAT EXAMPLES. The example above (childcare) is for FORMAT REFERENCE
ONLY — do NOT copy phrases, topics, specific details, or any content from it.
Your statement MUST address the actual question below using the actual opinions
and critiques provided. If you find yourself echoing words like "childcare",
"parental leave", "6 months", or "irrespective of gender", you are copying
the example — STOP and rewrite from scratch using the actual inputs below.
================================================================================

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
  prompt = f"""You are helping a group of citizens reach a consensus on a question they disagree about. Your job is to write the statement that the most people will rank first.

WHAT MAKES A WINNING STATEMENT:

- Front-load the verdict. State the group's position on the question in the first sentence. Do not describe the debate before taking a position.
- Write as the group, not about the group. Use "We believe...", "We should...", "In general, the group was opposed to...". Avoid descriptive framings like "some participants felt X while others felt Y" or "the group was divided" — those describe a debate instead of taking a stance.
- Borrow the specifics. Reuse concrete nouns, numbers, proposals, and institutions the participants named (e.g. "6 months", "HS2", "polling stations", "flexible working"). Do not restate them in generic terms.
- Propose, don't just oppose. When the group rejects an idea, pair it with a positive alternative or complementary proposal. A bare "no" rarely wins votes.
- Don't balance-wash dissent. If 4 of 5 agree, write the majority view in first person and compress the minority into a brief "however, we recognise that..." clause. Do not present minority and majority as equal alternatives.
- Be concrete, not abstract. Favor specific mechanisms ("polling stations open for longer") over vague principles ("improve voting access").
- Length follows substance. A crisp statement can beat a padded one. Include what the case needs, no more. Typically one substantial paragraph.
- The goal is first-place votes, not universal acceptance. Write to persuade. Clarity and stance beat diplomacy and hedging.
- Acknowledge minority concerns briefly rather than dismiss them, but still take a clear stance on the central question.
- Do NOT reference opinion or participant numbers in the final statement. Those belong only in the reasoning section.

Please think through this task step-by-step:

1. Identify the majority position on the central question.
2. List the concrete proposals, numbers, and named objects from the opinions that support that position.
3. Identify any minority concern strong enough to acknowledge.
4. Draft the statement: verdict first, reasons and borrowed specifics next, brief acknowledgment of minority concern if warranted, a concrete positive proposal where possible.

Provide your answer in the following format:

## Reasoning:
[Your step-by-step reasoning, referencing specific opinion numbers]

## Draft Consensus Statement:
[The draft consensus statement ONLY — written as the group's voice, with NO references to opinion numbers]

Example 1:

Question: Should the government provide universal free childcare from birth?

Individual Opinions:
Opinion Person 1: Free childcare is important but not from birth — babies benefit from a consistent primary caregiver. Offer universal paid parental leave from birth and universal free childcare from 6 months.
Opinion Person 2: Free childcare should be targeted at parents who can't afford it, not universal, and only from age three.
Opinion Person 3: Childcare costs are disproportionate to what one parent can earn. Drop-off and pick-up times limit careers.
Opinion Person 4: Parents often struggle between maternity leave ending and free childcare starting. Offer a choice of free childcare or a basic income.
Opinion Person 5: Childcare support helps parents and communities. Extended family isn't always available in the UK.

## Reasoning:
1. Opinions 1, 3, 4, and 5 support free childcare (majority); Opinion 2 wants it means-tested and limited to age 3+ (minority).
2. Most specific borrowable proposals: "6 months" and "universal paid parental leave from birth" (Opinion 1); the option of "free childcare or a basic income" (Opinion 4); the quality concern that childcare should support development, not just mind children (implied across 3, 5).
3. Opinion 2's means-testing is a minority view — brief acknowledgment, not equal weight.
4. Verdict: support free childcare, but not from birth. Borrow "6 months" and "universal paid parental leave" verbatim. Pair with the parent-choice proposal.

## Draft Consensus Statement:
In general, free childcare is a good thing, but it is important to consider how it is provided and for which age groups. We feel that it is important to offer support to parents in the form of parental leave, and that this should be available to both parents. We also feel that free childcare should be provided in a way that supports children's development and learning, not just as a childminding service. However, we do not feel that free childcare should be provided from birth, as it is important for babies to have a consistent primary caregiver in their early months. For this reason, we would support the government providing universal paid parental leave from birth, and universal free childcare from 6 months old. We would also offer parents the opportunity to either use free childcare between 6 months and 1 year, or to have paid parental leave for the same period.

Example 2:

Question: Should the government spend more on improving the railway network?

Individual Opinions:
Opinion Person 1: The government should spend more on the rail network. Delays come from problems with tracks, shortages of trains and staff, which puts people off using trains. For a greener future we need cars off roads and more people on public transport, and the only way to encourage that is a better service. Tickets are already expensive, so government support is needed.
Opinion Person 2: Trains are more energy efficient than other transport and improving rail would help cut carbon emissions and combat global warming. The government hasn't had to spend on roads the way railways need it, given how much road repair work is needed compared to rail.
Opinion Person 3: The government should spend more on rail, but not on HS2. Local railway services need improving so most people can use them. HS2 only benefits a few places. Italy has far better, cheaper rail, which encourages local transport and is good for the environment. Closed stations could be reopened.
Opinion Person 4: I'm not sure. It would mean raising taxes, and it would only be worth it if train travel were also made cheaper.

## Reasoning:
1. Opinions 1, 2, and 3 clearly support more spending on rail; Opinion 4 is conditional but would support it if travel were cheaper. That's a unanimous-or-near-unanimous majority.
2. Specific borrowable material: "HS2" from Opinion 3 (the group should redirect from it, not toward it); "energy efficient" and "pollution" from Opinion 2; "making rail travel cheaper" from Opinion 4; "repairing roads" / road condition contrast from Opinion 2.
3. No substantial minority to caveat — Opinion 4's conditionality is about price, which is addressed by the "cheaper travel" point.
4. Verdict: direct "The government should..." opening. Borrow HS2 and the road-repair contrast verbatim. Include the "cheaper travel" point as a concrete positive proposal.

## Draft Consensus Statement:
The government should spend more on improving the rail network. This is to encourage more people to use public transport, rather than using their cars. This will reduce congestion on the roads, and also reduce the amount of pollution caused by vehicles. The rail network is in desperate need of improvement, and this can only be done with more funding. The government should also look at making rail travel cheaper, to encourage more people to use it. It is a more energy efficient form of transport than cars and buses, and so it is better for the environment. This will also reduce the amount of money spent on repairing roads, which are in a poor state in many areas. It is a good idea to spend money on improving the rail network, rather than on the HS2, which will only benefit a few people.

It is CRITICAL to follow this format. Always include the "## Reasoning:" section followed by your explanation, then the "## Draft Consensus Statement:" section with ONLY the statement. The final statement must NOT contain any references like "(Opinion 1)" or "(Opinions 2, 3)".

================================================================================
END OF FORMAT EXAMPLES. The examples above (childcare, rail) are for FORMAT
REFERENCE ONLY — do NOT copy phrases, topics, specific details, or any content
from them. Your statement MUST address the actual question below using the
actual opinions provided. If you find yourself echoing words like "childcare",
"parental leave", "6 months", "rail network", or "HS2", you are copying the
example — STOP and rewrite from scratch using the actual inputs below.
================================================================================

Below you will find the question and the individual opinions of the participants.

Question: {question}

Individual Opinions:
"""

  for i, opinion in enumerate(opinions):
    prompt += f'Opinion Person {i+1}: {opinion}\n'

  return prompt.strip()


def _length_instruction(target_word_count: int | None) -> str:
  """Builds an explicit length-target clause to append to the prompt.

  The default prompts say "Length follows substance — typically one
  substantial paragraph". When the caller passes target_word_count, we want
  to override that with a concrete word target so long-form deliberations
  produce long-form statements that match the depth of the input opinions.
  """
  if target_word_count is None:
    return ''
  # The instruction is appended at the end so the model sees it last and
  # treats it as authoritative over the earlier "typically one paragraph"
  # guidance. ±20% gives the model room rather than forcing a brittle target.
  low = int(target_word_count * 0.8)
  high = int(target_word_count * 1.2)
  return (
      "\n\nLENGTH TARGET: Aim for approximately "
      f"{target_word_count} words in the final consensus statement "
      f"(roughly {low}-{high} words). The participants' opinions are "
      "long-form, so the statement should match that depth — develop the "
      "reasoning, name specifics, and engage substantively with the "
      "submitted views. Do not pad with filler; if you genuinely cannot "
      "reach the target without padding, prefer being shorter and tight."
  )


def _generate_prompt(
    question: str,
    opinions: Sequence[str],
    previous_winner: str | None = None,
    critiques: Sequence[str] | None = None,
    target_word_count: int | None = None,
) -> str:
  """Generates a prompt for the LLM."""
  if previous_winner is None:
    base = _generate_opinion_only_prompt(question, opinions)
  else:
    base = _generate_opinion_critique_prompt(
        question, opinions, previous_winner, critiques
    )
  return base + _length_instruction(target_word_count)


def _process_model_response(response: str) -> tuple[str, str]:
  """Processes the model's response, extracting the statement and explanation.

  Args:
      response: The raw model response.

  Returns:
      A tuple of (statement, explanation). If the response format is
      incorrect, returns ("", "INCORRECT_TEMPLATE").
  """
  # Try the canonical markdown format first (## headers with colons).
  match = re.search(
      r'##\s*Reasoning:\s*(.*?)##\s*(?:Draft|Revised)\s+Consensus Statement:\s*(.*?)(?:\n##|$)',
      response,
      re.DOTALL | re.IGNORECASE
  )
  if match:
    explanation = match.group(1).strip()
    statement = match.group(2).strip()
    if statement:
      return statement, explanation

  # Forgiving fallback: header markup the model sometimes drifts to —
  # bold (**Reasoning**), missing colons, "Final" instead of "Draft" /
  # "Revised", or an extra blank "##" line. We just need to find the
  # boundary between the reasoning block and the statement block.
  flexible_pattern = (
      r'(?:^|\n)\s*'
      r'(?:#{1,3}\s*|\*\*\s*)?'
      r'Reasoning\s*[:\*]*\s*'
      r'(?:#{0,3}\s*|\*\*)?\s*'
      r'(.*?)'
      r'(?:^|\n)\s*'
      r'(?:#{1,3}\s*|\*\*\s*)?'
      r'(?:Draft|Revised|Final)?\s*Consensus\s+Statement\s*[:\*]*\s*'
      r'(?:#{0,3}\s*|\*\*)?\s*'
      r'(.*?)(?:\n##|\n\*\*[A-Z]|$)'
  )
  match = re.search(flexible_pattern, response, re.DOTALL | re.IGNORECASE | re.MULTILINE)
  if match:
    explanation = match.group(1).strip()
    statement = match.group(2).strip()
    if statement:
      return statement, explanation

  # Fall back to old XML-like format for backward compatibility.
  match = re.search(
      r'<answer>\s*(.*?)\s*<sep>\s*(.*?)\s*</answer>', response, re.DOTALL
  )
  if match:
    explanation = match.group(1).strip()
    statement = match.group(2).strip()
    if statement:
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
      target_word_count: int | None = None,
  ) -> base_model.StatementResult:
    """Generates a statement (see base model)."""
    if num_retries_on_error is None:
      num_retries_on_error = 0
    else:
      if num_retries_on_error < 0:
        raise ValueError('num_retries_on_error must be None or at least 0.')
    prompt = _generate_prompt(
        question, opinions, previous_winner, critiques, target_word_count
    )
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


