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
import shap
import json


def load_shap_values(file_name):
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
  shap.initjs()


def graph_individual_example(data, index=0):
  return shap.force_plot(
      data['expected_values'],
      data['shap_values'][index, :],
      data['data'][index, :],
      feature_names=data['feature_names'])


def graph_summary_plot(data):
  return shap.summary_plot(
      data['shap_values'], data['data'], feature_names=data['feature_names'])
