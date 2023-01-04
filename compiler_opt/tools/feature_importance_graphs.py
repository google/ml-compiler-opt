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
"""A module that allows for easily graphing feature importance data in
notebooks"""

import numpy
import numpy.typing
import shap
import json

from typing import Dict, List, Union, Optional

DataType = Dict[str, Union[numpy.typing.ArrayLike, List[str]]]


def load_shap_values(file_name: str) -> DataType:
  """Loads a set of shap values created with the feature_importance.py script
  into a dictionary that can then be used for creating graphs

  Args:
    file_name: The name of the file in which the shap values are stored. What
      the --output_path flag was set to in the feature importance script.
  """
  with open(file_name, encoding='utf-8') as file_to_load:
    data = json.load(file_to_load)
    if data['expected_values'] is not list:
      data['expected_values'] = [data['expected_values']]
    return {
        'expected_values': numpy.asarray(data['expected_values']),
        'shap_values': numpy.asarray(data['shap_values']),
        'data': numpy.asarray(data['data']),
        'feature_names': data['feature_names']
    }


def init_shap_for_notebook():
  """Initalizes some JS code for interactive feature importance plots."""
  shap.initjs()


def graph_individual_example(data: DataType, index: Optional[int]):
  """Creates a force plot for an example

  Args:
    data: An object containing all the shap values and other information
      necessary to create the plot. Should be created with load_shap_values.
    index: The index of the example that you wish to plot.
  """
  return shap.force_plot(
      data['expected_values'],
      data['shap_values'][index, :],
      data['data'][index, :],
      feature_names=data['feature_names'])


def graph_summary_plot(data: DataType):
  """Creates a summary plot of the entire dataset given

  Args:
    data: An object containing all the shap values necessary to create the
      plot. Should come from load_shap_values
  """
  return shap.summary_plot(
      data['shap_values'], data['data'], feature_names=data['feature_names'])
