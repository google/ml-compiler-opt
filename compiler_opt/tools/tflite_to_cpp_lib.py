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
"""Library for converting between TFLite and C++."""
from __future__ import annotations

import os
import dataclasses
import subprocess
import pathlib
import re

from absl import logging

_UNUSED_TENSOR_NAME = '_UnusedTensorType'
_TFAGENTS_POLICY_NAME = 'action'
_MODEL_NAMESPACE = 'llvm::emitc::generated'


def _fmt_includes(includes):
  return '\n'.join([f'#include "{hdr}"' for hdr in includes]) + '\n'


@dataclasses.dataclass
class EmitCRuntime:
  """Holds the runtime buffers in memory."""

  # Maps [header name] -> [header contents]
  headers: dict[str, str]

  # Which is the primary head for the runtime? e.g., 'tosa.h'
  primary: str


def _load_emitc_runtime(path: str) -> EmitCRuntime:
  """Load the EmitC runtime from a given path."""
  headers = {}
  pathlist = pathlib.Path(path).glob('*.h')
  for p in pathlist:
    with open(p, 'rt', encoding='utf-8') as f:
      headers[p.name] = f.read()
  return EmitCRuntime(headers=headers, primary='tosa.h')


def _create_local_emitc_runtime(runtime: EmitCRuntime) -> str:
  """Create a "local" version of the EmitC runtime.

  The "local" version is analogous to a single-header version of the runtime,
  but intended to be put in a .cpp file. All relevant code is wrapped in an
  anonymous namespace in the .cpp file, so each model will have its own copy of
  the runtime.

  This function modifies the runtime in the following way:
    1) removes all macros aside from includes
    2) removes all comments/whitespace
    3) renames the Tensor type to _UNUSED_TENSOR_NAME. This is because the
    Tensor type is a part of the public API for the generated model, and to
    maintain uniformity between each generated model interface we will use a
    standard definition of Tensor in LLVM. This is the only part of the runtime
    which is shared between each generated model.

  This function depends on a particular implementation of the runtime which is
  prefered by mlcompileropt. To generalize this code, the function should
  topologically sort each header in the runtime by the inclusion ordering

  Args:
    runtime: the runtime to create a local version of.

  Returns:
    the contents of the local runtime as a string.
  """
  topsort_on_includes = [
      'utility.h',
      'types.h',
      'core_ops.h',
      'tensor.h',
      'tosa.h',
  ]
  assert set(topsort_on_includes).issubset(set(runtime.headers.keys()))
  # we don't currently support the eigen runtime, so set the file to zero
  runtime.headers['tosa_eigen.h'] = ''
  has_been_included = {key: False for key in topsort_on_includes}
  for key in topsort_on_includes:

    def on_match(m):
      group = m.group(1)
      if group not in topsort_on_includes or has_been_included[group]:
        return ''
      has_been_included[group] = True
      return runtime.headers[group]

    runtime.headers[key] = re.sub(
        r'#include "emitc/(\w+\.h)"',
        on_match,
        runtime.headers[key],
    )
  local_runtime = runtime.headers[runtime.primary]
  # Remove all comments, they just take up space
  local_runtime = re.sub(r'//.*', '', local_runtime)

  # Find any stdlib includes and store them
  stdlib_includes = re.findall(r'#include <(\w+)>', local_runtime)

  # Remove all the remaining macros
  local_runtime = re.sub(r'#.*', '', local_runtime)

  # Wrap the runtime in a local namespace to prevent ODR problems
  local_runtime = 'namespace {\n' + local_runtime + '\n}'

  # Reinsert the stdlib includes
  include_str = ('\n'.join([f'#include <{hdr}>' for hdr in stdlib_includes]) +
                 '\n')

  local_runtime = include_str + local_runtime

  # Rename the tensor type in the runtime, we will use LLVM's internal tensor
  # type so that the interface of each model is uniform. Theoretically, it
  # would be better to just remove this class, but renaming it is easier to
  # reason about.
  local_runtime = re.sub(r'class Tensor', f'class {_UNUSED_TENSOR_NAME}',
                         local_runtime)

  # We also need to rename the constructors of the class
  local_runtime = re.sub(r'Tensor\(', f'{_UNUSED_TENSOR_NAME}(', local_runtime)

  # Remove all empty newlines and return
  return '\n'.join(
      [l for l in local_runtime.splitlines() if (l and not l.isspace())])


@dataclasses.dataclass
class EmitCModel:
  # TODO: document this
  # TODO: get rid of cpp and hdr
  name: str
  cpp: str
  hdr: str


def _run_clang_format(buffer: str, clang_format_path: str,
                      clang_format_style: str) -> str:
  """Formats the given buffer and returns the result"""
  cmdline = [clang_format_path, f'--style={clang_format_style}']
  result = subprocess.run(
      cmdline, stdout=subprocess.PIPE, text=True, input=buffer, check=True)
  return result.stdout


def format_model(model: EmitCModel, clang_format_path: str,
                 clang_format_style: str) -> str:
  """Formats the given model and returns the result"""
  logging.info('Formatting the resulting model with style [%s].',
               clang_format_style)
  return dataclasses.replace(
      model,
      cpp=_run_clang_format(
          model.cpp,
          clang_format_path=clang_format_path,
          clang_format_style=clang_format_style,
      ),
      hdr=_run_clang_format(
          model.hdr,
          clang_format_path=clang_format_path,
          clang_format_style=clang_format_style,
      ),
  )


def get_model_cpp_path(model: EmitCModel, root: str) -> str:
  return os.path.join(root, model.name + '.emitc.cpp')


def get_model_hdr_path(model: EmitCModel, root: str) -> str:
  return os.path.join(root, model.name + '.emitc.h')


def tflite_to_tosa(tflite_path: str,
                   iree_import_tflite_path: str,
                   *,
                   convert_i48=True) -> str:
  """Converts TFLite to TOSA MLIR."""
  logging.info('Converting the TFLite model to TOSA MLIR.')
  cmdline = [
      iree_import_tflite_path,
      '-o',
      '-',
      tflite_path,
      '--output-format=mlir-ir',
  ]
  result = subprocess.run(
      cmdline, stdout=subprocess.PIPE, text=True, check=True)
  if convert_i48:
    return re.sub(r'i48', 'i64', result.stdout)
  return result.stdout


def tosa_to_emitc_mlir(tosa: str, emitc_opt_path: str) -> str:
  """Converts TOSA MLIR to EmitC MLIR using emitc-opt."""
  logging.info('Converting the TOSA MLIR to EmitC MLIR.')
  cmdline = [emitc_opt_path, '--convert-tosa-to-emitc', '-o', '-', '-']
  result = subprocess.run(
      cmdline, stdout=subprocess.PIPE, text=True, input=tosa, check=True)
  return result.stdout


def emitc_mlir_to_cpp(
    emitc_mlir: str,
    mlir_translate_path: str,
    name: str,
    base_class: str,
) -> EmitCModel:
  """Converts EmitC MLIR to C++ files using mlir-translate."""
  logging.info('Converting the EmitC MLIR to C++.')

  def _get_cmdline(kind: str):
    return [
        mlir_translate_path,
        '-mlir-to-cpp',
        '--emit-cpp-kind=stateful',
        '--emit-cpp-arg-name-attr=tf_saved_model.index_path',
        f'--emit-cpp-model-name={name}',
        f'--emit-cpp-base-class={base_class}',
        f'--emit-cpp-file-kind={kind}',
        f'--emit-cpp-only-one-fn={_TFAGENTS_POLICY_NAME}',
        '-o',
        '-',
        '-',
    ]

  result_cpp = subprocess.run(
      _get_cmdline('cpp'),
      stdout=subprocess.PIPE,
      text=True,
      input=emitc_mlir,
      check=True,
  ).stdout
  result_hdr = subprocess.run(
      _get_cmdline('header'),
      stdout=subprocess.PIPE,
      text=True,
      input=emitc_mlir,
      check=True,
  ).stdout

  # Wrap results in namespaces
  result_cpp = f'namespace {_MODEL_NAMESPACE} {{' + '\n' + result_cpp + '}\n'
  result_hdr = f'namespace {_MODEL_NAMESPACE} {{' + '\n' + result_hdr + '}\n'

  return EmitCModel(cpp=result_cpp, hdr=result_hdr, name=name)


def embed_runtime(
    model: EmitCModel,
    runtime_path: str,
) -> EmitCModel:
  """Embed the emitc runtime in the model.cpp file.

  This also:
    1) renames any types that are coming from LLVM instead of the embedded
       runtime, and
    2) includes all required headers

  Args:
    model: the model which we are embedding the runtime into.
    runtime_path: path to the emitc runtime to embed.

  Returns:
    the new model
  """
  logging.info('Embedding the EmitC runtime in the generated model.')

  runtime = _load_emitc_runtime(runtime_path)
  local_runtime = _create_local_emitc_runtime(runtime)

  new_cpp = local_runtime + model.cpp

  # Rename any uses of the Tensor template type to the fully qualified LLVM name
  # This regex uses a negative character lookbehind, so:
  #   `(Tensor<` and ` Tensor<`
  # both match, but
  #   `IsTensor<`
  # does not. the latter appears in the runtime, which we don't want to replace
  new_cpp = re.sub(r'(?<![A-Za-z])Tensor<', r'::llvm::emitc::Tensor<', new_cpp)
  new_hdr = re.sub(r'(?<![A-Za-z])Tensor<', r'::llvm::emitc::Tensor<',
                   model.hdr)

  # We also need to fully-qualify the references to emitc:: because the emitc
  # namespace is ambiguous in the file. This uses a similar lookbehind to avoid
  # replacing `llvm::emitc` which is what makes the namespace ambiguous
  new_cpp = re.sub(r'(?<!llvm::)emitc::', r'::emitc::', new_cpp)

  # Add necessary includes to both files
  cpp_includes = ['llvm/Analysis/EmitCTensor.h', f'{model.name}.emitc.h']
  hdr_includes = ['llvm/Analysis/EmitCTensor.h']

  new_cpp = _fmt_includes(cpp_includes) + new_cpp
  new_hdr = _fmt_includes(hdr_includes) + new_hdr

  return dataclasses.replace(model, cpp=new_cpp, hdr=new_hdr)


def add_additional_headers(model: EmitCModel, additional_headers: list[str]):
  include_str = _fmt_includes(additional_headers)
  new_hdr = include_str + model.hdr
  return dataclasses.replace(model, hdr=new_hdr)


def print_llvm_registration_handle(model: EmitCModel, base_class: str):
  """Prints LLVM model registration code.

  This handle automatically adds the model to a global registry of models that
  are available in LLVM, so all that needs to be done to integrate the model in
  LLVM is link the .cpp with the required binary.
  """
  registration_msg = f"""
{'*'*60}
To register the generated model in LLVM, please include the
generated header and copy the following code into a .cpp file:

REGISTER_EMITC_MODEL({base_class}, {model.name});

Note the generated .cpp file must include the line at least once:

#include "llvm/Analysis/EmitCModelRegistry.h"
{'*'*60}
"""
  logging.info(registration_msg)
