# coding=utf-8
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
r"""Merge multiple best trajectory repo into one."""

import json

from absl import app
from absl import flags
from absl import logging

from compiler_opt.rl import best_trajectory

_BEST_TRAJECTORY_PATHS = flags.DEFINE_multi_string(
    'best_trajectory_paths', '',
    'best trajectory repo dump to be merged in json format.')
_OUTPUT_JSON_PATH = flags.DEFINE_string(
    'output_json_path', '',
    'output path of the merged best trajectory repo in json format if given.')
_OUTPUT_CSV_PATH = flags.DEFINE_string(
    'output_csv_path', '',
    'output path of the merged best trajectory repo in csv format if given.')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # we don't use action_name in the merging process here, so just set it to
  # empty string.
  merged_best_trajectory_repo = best_trajectory.BestTrajectoryRepo(
      action_name='')

  for path in _BEST_TRAJECTORY_PATHS.value:
    logging.info('merging repo: %s', path)
    tmp = best_trajectory.BestTrajectoryRepo(action_name='')
    # The json file is broken sometimes.
    # Open issue: https://github.com/google/ml-compiler-opt/issues/163
    try:
      tmp.load_from_json_file(path)
      merged_best_trajectory_repo.combine_with_other_repo(tmp)
    except json.decoder.JSONDecodeError:
      logging.error('failed to load input repo: %s', path)

  if _OUTPUT_JSON_PATH.value:
    tmp.sink_to_json_file(_OUTPUT_JSON_PATH.value)

  if _OUTPUT_CSV_PATH.value:
    tmp.sink_to_csv_file(_OUTPUT_CSV_PATH.value)


if __name__ == '__main__':
  app.run(main)
