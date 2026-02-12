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
"""Manage baseline scores (the reference from which we form a reward)"""

from typing import Any, Generic, TypeVar
from collections.abc import Callable

T = TypeVar("T")


class BaselineCache(Generic[T]):
  """Manages a cache of baseline scores."""

  def __init__(self, *, get_scores: Callable[[list[T]], list[float]],
               get_key: Callable[[T], Any]):
    """Constructor.

        Args:
            get_scores: A callable that returns the scores for a batch of items.
            The callable is responsible for timely completion. It must not
            raise, and it must return results in the order of the items
            provided. A None value is expected for items that could not produce
            a value.
            get_key: A callable that returns the key for an item.
        """
    self._get_scores = get_scores
    self._get_key = get_key
    self._cache = {}

  def get_score(self, items: list[T | None]):
    """Get the scores for a batch of items.
        The scores are returned in the same order as the provided items. A None
        result indicates the score could not be obtained.

        Args:
            items: A list of items to get scores for.
        """
    todo = [i for i in items if self._get_key(i) not in self._cache]
    scores = self._get_scores(todo)
    if len(scores) != len(todo):
      raise ValueError(
          "got a different number of results for the requested items")
    for i, s in zip(todo, scores):
      self._cache[self._get_key(i)] = s
    return [self._cache[self._get_key(i)] for i in items]

  def get_cache(self):
    """Intended for testing."""
    return self._cache
