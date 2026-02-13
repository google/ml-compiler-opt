# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for BlackboxEvaluator."""

import concurrent.futures

from absl.testing import absltest

from compiler_opt.distributed.local import local_worker_manager
from compiler_opt.distributed import buffered_scheduler, worker
from compiler_opt import baseline_cache


# this mocks es - style workers, that may take a list of one or more items to
# compile. The list is assumed to be strings representing ints
class MockWorker(worker.Worker):

  def do_some_work(self, *, items: list[str]):
    return sum(int(x) for x in items)


class BaselineCacheTest(absltest.TestCase):
  """Tests for BlackboxEvaluator."""

  def test_simple(self):
    mock = {"a": 1, "b": 2, "c": 3}
    score_asked_for = []

    def track_score(lst):
      score_asked_for.extend(lst)
      return [mock[k] if k in mock else None for k in lst]

    cache = baseline_cache.BaselineCache(
        get_scores=track_score, get_key=lambda x: x)
    self.assertEmpty(cache.get_cache())
    self.assertEqual(cache.get_score(["c", "b"]), [3, 2])
    self.assertDictEqual(cache.get_cache(), {"b": 2, "c": 3})
    self.assertListEqual(score_asked_for, ["c", "b"])
    score_asked_for.clear()

    self.assertEqual(cache.get_score(["c", "b"]), [3, 2])
    self.assertListEqual(score_asked_for, [])
    self.assertEqual(cache.get_score(["a", "c", "b"]), [1, 3, 2])
    self.assertListEqual(score_asked_for, ["a"])
    score_asked_for.clear()

    self.assertEqual(cache.get_score(["a", "n", "c", "b"]), [1, None, 3, 2])
    self.assertListEqual(score_asked_for, ["n"])
    score_asked_for.clear()

    self.assertEqual(cache.get_score(["a", "n", "c", "b"]), [1, None, 3, 2])
    self.assertListEqual(score_asked_for, [])

  def test_with_workers(self):
    with local_worker_manager.LocalWorkerPoolManager(
        worker_class=MockWorker, count=4) as lwm:

      score_asked_for = []

      def get_scores(items: list[str]):
        score_asked_for.extend(items)
        _, futures = buffered_scheduler.schedule_on_worker_pool(
            action=lambda w, x: w.do_some_work(items=[x]),
            jobs=items,
            worker_pool=lwm)
        concurrent.futures.wait(
            futures, return_when=concurrent.futures.ALL_COMPLETED)
        return [f.result() if f.exception() is None else None for f in futures]

      cache = baseline_cache.BaselineCache(
          get_key=lambda x: x, get_scores=get_scores)
      self.assertEmpty(cache.get_cache())
      self.assertEqual(cache.get_score(["4", "2"]), [4, 2])
      self.assertListEqual(score_asked_for, ["4", "2"])
      self.assertDictEqual(cache.get_cache(), {"4": 4, "2": 2})
      score_asked_for.clear()

      self.assertEqual(cache.get_score(["4", "2"]), [4, 2])
      self.assertListEqual(score_asked_for, [])
