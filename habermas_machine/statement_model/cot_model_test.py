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

from absl.testing import absltest
from absl.testing import parameterized

from habermas_machine.llm_client import base_client
from habermas_machine.llm_client import mock_client
from habermas_machine.statement_model import cot_model


class _PromptCapturingClient(base_client.LLMClient):
  """LLM client that records every prompt it sees and returns a fixed reply.

  Used to verify that target_word_count plumbed through generate_statement
  ends up in the actual prompt sent to the model.
  """

  def __init__(self, response: str):
    self._response = response
    self.last_prompt: str | None = None

  def sample_text(self, prompt, *, max_tokens=base_client.DEFAULT_MAX_TOKENS,
                  terminators=base_client.DEFAULT_TERMINATORS,
                  temperature=base_client.DEFAULT_TEMPERATURE,
                  timeout=base_client.DEFAULT_TIMEOUT_SECONDS,
                  seed=None):
    del max_tokens, terminators, temperature, timeout, seed
    self.last_prompt = prompt
    return self._response


class COTModelTest(parameterized.TestCase):

  def test_generate_opinion_only_prompt(self):
    """Tests the _generate_opinion_only_prompt method."""
    question = "What is the best way to improve public transport?"
    opinions = [
        "Invest in more buses.",
        "Reduce fares.",
        "Improve the train network."
    ]
    prompt = cot_model._generate_opinion_only_prompt(question, opinions)
    self.assertIn(question, prompt)
    for opinion in opinions:
      self.assertIn(opinion, prompt)
    self.assertIn("## Reasoning:", prompt)
    self.assertIn("## Draft Consensus Statement:", prompt)

  def test_generate_opinion_critique_prompt(self):
    """Tests the _generate_opinion_critique_prompt method."""
    question = "How can we make our city greener?"
    opinions = [
        "Plant more trees.",
        "Create more parks.",
        "Reduce car usage."
    ]
    previous_winner = "We should plant more trees in the city center."
    critiques = [
        "Planting trees is not enough.",
        "We need more parks, not just trees."
    ]
    prompt = cot_model._generate_opinion_critique_prompt(
        question, opinions, previous_winner, critiques
    )
    self.assertIn(question, prompt)
    for opinion in opinions:
      self.assertIn(opinion, prompt)
    self.assertIn(previous_winner, prompt)
    for critique in critiques:
      self.assertIn(critique, prompt)
    self.assertIn("## Reasoning:", prompt)
    self.assertIn("## Revised Consensus Statement:", prompt)

  def test_generate_prompt(self):
    """Tests the _generate_prompt method."""
    question = "What is the meaning of life?"
    opinions = ["42", "To be happy", "To help others"]

    # Test with only opinions
    prompt = cot_model._generate_prompt(question, opinions)
    self.assertEqual(
        prompt, cot_model._generate_opinion_only_prompt(question, opinions)
    )

    # Test with previous winner and critiques
    previous_winner = "42 is the answer."
    critiques = ["That's not helpful", "What about happiness?"]
    prompt = cot_model._generate_prompt(
        question, opinions, previous_winner, critiques
    )
    self.assertEqual(
        prompt,
        cot_model._generate_opinion_critique_prompt(
            question, opinions, previous_winner, critiques
        ),
    )

  def test_process_model_response(self):
    """Tests the _process_model_response method."""
    # Test correct format
    response = (
        "<answer>This is the explanation.\n<sep>\nThis is the"
        " statement.</answer>"
    )
    statement, explanation = cot_model._process_model_response(response)
    self.assertEqual(statement, "This is the statement.")
    self.assertEqual(explanation, "This is the explanation.")

    # Test incorrect format
    response = "This is just some text."
    statement, explanation = cot_model._process_model_response(response)
    self.assertEqual(statement, "")
    self.assertEqual(explanation, "INCORRECT_TEMPLATE")

  def test_process_model_response_canonical_markdown(self):
    """The default ## Reasoning: / ## Draft Consensus Statement: format."""
    response = (
        "## Reasoning:\n"
        "Some reasoning here.\n"
        "## Draft Consensus Statement:\n"
        "We believe X."
    )
    statement, explanation = cot_model._process_model_response(response)
    self.assertEqual(statement, "We believe X.")
    self.assertEqual(explanation, "Some reasoning here.")

  @parameterized.named_parameters(
      ("bold_no_colon",
       "**Reasoning**\nThe reasoning is sound.\n\n"
       "**Draft Consensus Statement**\nWe support the proposal."),
      ("h3_with_colon",
       "### Reasoning:\nA detailed analysis.\n\n"
       "### Draft Consensus Statement:\nWe support the proposal."),
      ("revised_variant",
       "## Reasoning:\nWe revisited the draft.\n\n"
       "## Revised Consensus Statement:\nWe support the proposal."),
      ("final_variant",
       "## Reasoning:\nWe revisited the draft.\n\n"
       "## Final Consensus Statement:\nWe support the proposal."),
  )
  def test_process_model_response_forgives_format_drift(self, response):
    """Common format-drift variants should still parse, not return empty."""
    statement, explanation = cot_model._process_model_response(response)
    self.assertEqual(statement, "We support the proposal.")
    self.assertNotEmpty(explanation)
    self.assertNotIn("INCORRECT", explanation)

  @parameterized.named_parameters(
      ("opinion_only", None, None, "Draft Consensus Statement"),
      ("with_critique", "Previous winner.", ["c1", "c2"],
       "Revised Consensus Statement"),
  )
  def test_final_format_reminder_appended_after_inputs(
      self, previous_winner, critiques, expected_header
  ):
    """The format reminder must appear AFTER the opinions/critiques."""
    prompt = cot_model._generate_prompt(
        "Q?", ["op1", "op2"], previous_winner, critiques
    )
    # Reminder block is present.
    self.assertIn("NOW PRODUCE YOUR OUTPUT", prompt)
    self.assertIn(f"## {expected_header}:", prompt)
    self.assertIn('START YOUR RESPONSE WITH "## Reasoning:"', prompt)
    # And it really is at the end, after the inputs — the reminder
    # block must come strictly after the last opinion/critique line.
    last_input_marker = (
        prompt.rfind("Critique Person 2:") if critiques
        else prompt.rfind("Opinion Person 2:")
    )
    reminder_position = prompt.find("NOW PRODUCE YOUR OUTPUT")
    self.assertGreater(reminder_position, last_input_marker)

  def test_process_model_response_rejects_empty_statement(self):
    """A header with no content under it must NOT pass as a valid statement."""
    response = (
        "## Reasoning:\nLots of reasoning.\n"
        "## Draft Consensus Statement:\n"
        # Empty body — no statement.
    )
    statement, explanation = cot_model._process_model_response(response)
    self.assertEqual(statement, "")
    self.assertEqual(explanation, "INCORRECT_TEMPLATE")

  def test_generate_statement(self):
    """Tests the generate_statement method."""

    model = cot_model.COTModel()
    llm_client = mock_client.MockClient(
        response="<answer>Mock explanation<sep>Mock statement</answer>")
    question = "Test question?"
    opinions = ["Opinion 1", "Opinion 2"]
    statement, explanation = model.generate_statement(
        llm_client, question, opinions
    )
    self.assertEqual(statement, "Mock statement")
    self.assertEqual(explanation, "Mock explanation")

  def test_length_instruction_returns_empty_when_target_is_none(self):
    self.assertEqual(cot_model._length_instruction(None), "")

  def test_length_instruction_contains_target_and_range(self):
    instruction = cot_model._length_instruction(850)
    # Target word count appears verbatim.
    self.assertIn("850 words", instruction)
    # The ±20% range is rendered (low=680, high=1020).
    self.assertIn("680", instruction)
    self.assertIn("1020", instruction)
    # The directive uses an explicit marker so the model can find it
    # under the earlier "typically one paragraph" guidance.
    self.assertIn("LENGTH TARGET", instruction)
    # The "prefer shorter than padded" caveat is present so we don't
    # encourage the model to pad.
    self.assertIn("padding", instruction)

  @parameterized.named_parameters(
      ("opinion_only", None, None),
      ("with_critique", "Previous winner.", ["Critique 1.", "Critique 2."]),
  )
  def test_generate_prompt_appends_length_instruction(
      self, previous_winner, critiques
  ):
    """The length instruction shows up at the END of the generated prompt."""
    question = "Q?"
    opinions = ["op A", "op B"]

    prompt_without = cot_model._generate_prompt(
        question, opinions, previous_winner, critiques
    )
    prompt_with = cot_model._generate_prompt(
        question, opinions, previous_winner, critiques,
        target_word_count=400,
    )

    # No target -> no length instruction text.
    self.assertNotIn("LENGTH TARGET", prompt_without)

    # Target -> instruction appended at the end (so the model treats it
    # as authoritative over the upstream "typically one paragraph"
    # guidance baked into the few-shot examples).
    self.assertIn("LENGTH TARGET", prompt_with)
    self.assertIn("400 words", prompt_with)
    self.assertTrue(
        prompt_with.startswith(prompt_without),
        msg="Length instruction should be appended, not interleaved.",
    )

  def test_generate_statement_threads_target_word_count_into_prompt(self):
    """COTModel.generate_statement must forward target_word_count."""
    model = cot_model.COTModel()
    spy = _PromptCapturingClient(
        response="<answer>e<sep>s</answer>"
    )

    model.generate_statement(
        spy,
        question="Q?",
        opinions=["op"],
        target_word_count=350,
    )

    self.assertIsNotNone(spy.last_prompt)
    self.assertIn("LENGTH TARGET", spy.last_prompt)
    self.assertIn("350 words", spy.last_prompt)

  def test_generate_statement_default_omits_length_instruction(self):
    """Sanity check: the instruction is only added when explicitly requested."""
    model = cot_model.COTModel()
    spy = _PromptCapturingClient(
        response="<answer>e<sep>s</answer>"
    )

    model.generate_statement(spy, question="Q?", opinions=["op"])

    self.assertIsNotNone(spy.last_prompt)
    self.assertNotIn("LENGTH TARGET", spy.last_prompt)


if __name__ == "__main__":
  absltest.main()
