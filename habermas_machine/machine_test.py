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

from collections.abc import Sequence

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from habermas_machine import machine
from habermas_machine import types
from habermas_machine.llm_client import base_client
from habermas_machine.reward_model import base_model as reward_base
from habermas_machine.social_choice import utils as sc_utils


class _FailingRankingModel(reward_base.BaseRankingModel):
  """Test reward model that fails (returns None) for specified opinions.

  Used to exercise the drop-failed-citizens path in HabermasMachine —
  any opinion in `failing_opinions` produces a None ranking, simulating
  the LLM returning a partial/malformed answer that can't be parsed.
  """

  def __init__(self, failing_opinions: set[str]):
    self._failing = failing_opinions

  def predict_ranking(
      self,
      llm_client: base_client.LLMClient,
      question: str,
      opinion: str,
      statements: Sequence[str],
      previous_winner: str | None = None,
      critique: str | None = None,
      seed: int | None = None,
      num_retries_on_error: int = 1,
  ) -> reward_base.RankingResult:
    del (llm_client, question, previous_winner, critique, seed,
         num_retries_on_error)
    if opinion in self._failing:
      return reward_base.RankingResult(
          ranking=None,
          explanation=f'INCORRECT_RANKING_LENGTH (synthetic) for {opinion!r}',
      )
    # A trivial deterministic ranking: every candidate ranked equally.
    return reward_base.RankingResult(
        ranking=np.zeros(len(statements), dtype=int),
        explanation=f'ok for {opinion!r}',
    )


class HabermasMachineTest(parameterized.TestCase):

  def test_habermas_machine(self):
    """Tests the habermas machine."""
    num_citizens = 2
    num_candidates = 3  # Generate only one candidate for easier checking
    question = 'What is the meaning of life?'
    opinions = ['42', 'To be happy.']
    critiques = ['I disagree.', 'I agree.']
    hm = machine.HabermasMachine(
        question=question,
        statement_client=types.LLMCLient.MOCK.get_client('mock_url'),
        reward_client=types.LLMCLient.MOCK.get_client('mock_url'),
        statement_model=types.StatementModel.MOCK.get_model(),
        reward_model=types.RewardModel.MOCK.get_model(),
        social_choice_method=types.RankAggregation.SCHULZE.get_method(
            tie_breaking_method=sc_utils.TieBreakingMethod.TIES_ALLOWED
        ),
        num_candidates=num_candidates,
        num_citizens=num_citizens,
        seed=0,
    )

    # Test initial state.
    self.assertEqual(hm._round, 0)
    self.assertEmpty(hm._opinions)
    self.assertEmpty(hm._critiques)
    self.assertEmpty(hm._previous_winners)
    self.assertEmpty(hm._previous_candidates)
    self.assertEqual(hm._num_citizens, num_citizens)
    self.assertEqual(hm._num_candidates, num_candidates)
    self.assertEqual(hm._verbose, False)
    self.assertEqual(hm._question, question)

    # Test overwrite previous winner.
    with self.assertRaises(ValueError):
      hm.overwrite_previous_winner('winner1')

    # Test opinion round.
    winner, sorted_statements = hm.mediate(opinions)

    self.assertEqual(hm._round, 1)
    self.assertEqual(hm._question, question)
    self.assertSequenceEqual(hm._opinions, opinions)
    self.assertEmpty(hm._critiques)
    self.assertSequenceEqual(hm._previous_winners, [winner])
    self.assertSequenceEqual(hm._previous_candidates, [sorted_statements])
    self.assertIn(winner, sorted_statements)
    for statement in sorted_statements:
      self.assertIn(question, statement)
      for opinion in opinions:
        self.assertIn(opinion, statement)

    # Test critique round.
    winner2, sorted_statements2 = hm.mediate(critiques)

    self.assertEqual(hm._round, 2)
    self.assertEqual(hm._question, question)
    self.assertSequenceEqual(hm._opinions, opinions)
    self.assertSequenceEqual(hm._critiques, [critiques])
    self.assertSequenceEqual(hm._previous_winners, [winner, winner2])
    self.assertSequenceEqual(
        hm._previous_candidates, [sorted_statements, sorted_statements2]
    )
    self.assertIn(winner2, sorted_statements2)
    for statement in sorted_statements2:
      self.assertIn(question, statement)
      for opinion in opinions:
        self.assertIn(opinion, statement)
      for critique in critiques:
        self.assertIn(critique, statement)

    # Test overwrite previous winner.
    hm.overwrite_previous_winner('winner3')
    self.assertEqual(hm._previous_winners[-1], 'winner3')

  def test_wrong_number_of_opinions(self):
    num_citizens = 2
    hm = machine.HabermasMachine(
        question='Question?',
        statement_client=types.LLMCLient.MOCK.get_client('mock_url'),
        reward_client=types.LLMCLient.MOCK.get_client('mock_url'),
        statement_model=types.StatementModel.MOCK.get_model(),
        reward_model=types.RewardModel.MOCK.get_model(),
        social_choice_method=types.RankAggregation.SCHULZE.get_method(
            tie_breaking_method=sc_utils.TieBreakingMethod.TIES_ALLOWED
        ),
        num_citizens=num_citizens,
        seed=0,
    )
    with self.assertRaises(ValueError):
      _, _ = hm.mediate(['opinion1'])

  def test_wrong_number_of_critiques(self):
    num_citizens = 2
    hm = machine.HabermasMachine(
        question='Question?',
        statement_client=types.LLMCLient.MOCK.get_client('mock_url'),
        reward_client=types.LLMCLient.MOCK.get_client('mock_url'),
        statement_model=types.StatementModel.MOCK.get_model(),
        reward_model=types.RewardModel.MOCK.get_model(),
        social_choice_method=types.RankAggregation.SCHULZE.get_method(
            tie_breaking_method=sc_utils.TieBreakingMethod.TIES_ALLOWED
        ),
        num_citizens=num_citizens,
        seed=0,
    )
    _, _ = hm.mediate(['opinion1', 'opinion2'])  # Opinion round.
    with self.assertRaises(ValueError):
      hm.mediate(['critique1', 'critique2', 'critique3'])

  def _build_for_parallel(self, max_workers: int) -> machine.HabermasMachine:
    return machine.HabermasMachine(
        question='Q?',
        statement_client=types.LLMCLient.MOCK.get_client('mock_url'),
        reward_client=types.LLMCLient.MOCK.get_client('mock_url'),
        statement_model=types.StatementModel.MOCK.get_model(),
        reward_model=types.RewardModel.MOCK.get_model(),
        social_choice_method=types.RankAggregation.SCHULZE.get_method(
            tie_breaking_method=sc_utils.TieBreakingMethod.TIES_ALLOWED
        ),
        num_candidates=4,
        num_citizens=3,
        seed=42,
        max_workers=max_workers,
    )

  def test_parallel_rankings_match_serial(self):
    """max_workers > 1 must produce identical output for a given seed."""
    opinions = ['o1', 'o2', 'o3']
    critiques = ['c1', 'c2', 'c3']

    serial = self._build_for_parallel(max_workers=1)
    w_s1, c_s1 = serial.mediate(opinions)
    w_s2, c_s2 = serial.mediate(critiques)

    parallel = self._build_for_parallel(max_workers=3)
    w_p1, c_p1 = parallel.mediate(opinions)
    w_p2, c_p2 = parallel.mediate(critiques)

    self.assertEqual(w_s1, w_p1)
    self.assertSequenceEqual(c_s1, c_p1)
    self.assertEqual(w_s2, w_p2)
    self.assertSequenceEqual(c_s2, c_p2)

  def test_invalid_max_workers(self):
    with self.assertRaises(ValueError):
      self._build_for_parallel(max_workers=0)

  def _build_with_failing_ranker(
      self, failing_opinions: set[str], num_citizens: int
  ) -> machine.HabermasMachine:
    return machine.HabermasMachine(
        question='Q?',
        statement_client=types.LLMCLient.MOCK.get_client('mock_url'),
        reward_client=types.LLMCLient.MOCK.get_client('mock_url'),
        statement_model=types.StatementModel.MOCK.get_model(),
        reward_model=_FailingRankingModel(failing_opinions),
        social_choice_method=types.RankAggregation.SCHULZE.get_method(
            tie_breaking_method=sc_utils.TieBreakingMethod.TIES_ALLOWED
        ),
        num_candidates=4,
        num_citizens=num_citizens,
        seed=0,
    )

  def test_failed_citizens_are_dropped_not_raised(self):
    """A few citizens failing to rank should not abort the whole round."""
    opinions = ['op_a', 'op_b', 'op_c', 'op_d', 'op_e']
    # 1 of 5 fails — well under the >50% drop-loudly threshold.
    hm = self._build_with_failing_ranker({'op_b'}, num_citizens=5)

    winner, sorted_statements = hm.mediate(opinions)

    self.assertIsNotNone(winner)
    self.assertNotEmpty(sorted_statements)
    # The dropped citizen is exposed (1-indexed).
    self.assertEqual(hm.last_round_dropped_citizens, [2])

  def test_no_drops_means_empty_property(self):
    """Property stays empty when every citizen produces a valid ranking."""
    opinions = ['op_a', 'op_b', 'op_c']
    hm = self._build_with_failing_ranker(set(), num_citizens=3)

    hm.mediate(opinions)

    self.assertEqual(hm.last_round_dropped_citizens, [])

  def test_too_many_drops_raises(self):
    """If majority of citizens fail, raise so the caller knows it's broken."""
    opinions = ['op_a', 'op_b', 'op_c', 'op_d', 'op_e']
    # 3 of 5 fail — over the 50% threshold.
    hm = self._build_with_failing_ranker(
        {'op_a', 'op_b', 'op_c'}, num_citizens=5
    )

    with self.assertRaisesRegex(ValueError, 'Too many citizens'):
      hm.mediate(opinions)

  def test_dropped_property_resets_per_round(self):
    """last_round_dropped_citizens should reflect only the most recent round."""
    opinions = ['op_a', 'op_b', 'op_c', 'op_d', 'op_e']
    hm = self._build_with_failing_ranker({'op_c'}, num_citizens=5)
    hm.mediate(opinions)
    self.assertEqual(hm.last_round_dropped_citizens, [3])

    # Critique round, no failures this time. Reuse the same hm; drops
    # should clear back to [].
    hm._reward_model = _FailingRankingModel(set())  # swap in clean model
    critiques = ['c_a', 'c_b', 'c_c', 'c_d', 'c_e']
    hm.mediate(critiques)
    self.assertEqual(hm.last_round_dropped_citizens, [])


if __name__ == '__main__':
  absltest.main()
