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
"""Script for converting between TFLite and C++."""
from absl import app
from absl import flags
from absl import logging

from compiler_opt.tools import tflite_to_cpp_lib

flags.DEFINE_string('input', None,
                    'Input, which should be a path to a tflite model')
flags.mark_flag_as_required('input')

flags.DEFINE_string('output_dir', None,
                    'Output directory for the generated files')
flags.mark_flag_as_required('output_dir')

flags.DEFINE_string(
    'name',
    None,
    ('Name to use for the model. This will be in the filenames and also will'
     ' be used to identify the model within LLVM. This should be unique'
     ' between models'),
)
flags.mark_flag_as_required('name')

flags.DEFINE_string(
    'base_class',
    None,
    ('Base class to use for the generated model. This is used when'
     ' registering the model in LLVM. This should be a fully-qualified name,'
     ' e.g. ::llvm::MLInlineOzEmitCModel'),
)
flags.mark_flag_as_required('base_class')

flags.DEFINE_multi_string(
    'additional_headers',
    None,
    ('Additional headers to include for the model, for instance the header'
     ' definining the base class. Should be of the form'
     ' --additional_headers="llvm/Analysis/MyHeader.h"'),
)

flags.DEFINE_string(
    'iree_import_tflite_path',
    None,
    'Path to the iree-import-tflite binary from iree repository',
)
flags.mark_flag_as_required('iree_import_tflite_path')

flags.DEFINE_string(
    'emitc_opt_path',
    None,
    'Path to the emitc-opt binary from the emitc repository',
)
flags.mark_flag_as_required('emitc_opt_path')

flags.DEFINE_string(
    'mlir_translate_path',
    None,
    'Path to the mlir-translate binary from the llvm repository',
)
flags.mark_flag_as_required('mlir_translate_path')

flags.DEFINE_string(
    'emitc_runtime_path',
    None,
    'Path to the emitc runtime to embed in the generated c++ model',
)
flags.mark_flag_as_required('emitc_runtime_path')

flags.DEFINE_string(
    'clang_format_path',
    None,
    ('(Optional) path to clang-format binary to use to format the resulting'
     ' files'),
)
flags.DEFINE_string(
    'clang_format_style',
    'llvm',
    'Style argument to use for clang format',
)

FLAGS = flags.FLAGS


def main(argv):
  del argv
  logging.info('Beginning conversion pipeline.')
  tosa = tflite_to_cpp_lib.tflite_to_tosa(
      tflite_path=FLAGS.input,
      iree_import_tflite_path=FLAGS.iree_import_tflite_path,
  )
  emitc_mlir = tflite_to_cpp_lib.tosa_to_emitc_mlir(
      tosa=tosa, emitc_opt_path=FLAGS.emitc_opt_path)
  model = tflite_to_cpp_lib.emitc_mlir_to_cpp(
      emitc_mlir=emitc_mlir,
      mlir_translate_path=FLAGS.mlir_translate_path,
      name=FLAGS.name,
      base_class=FLAGS.base_class,
  )
  model = tflite_to_cpp_lib.embed_runtime(
      model=model,
      runtime_path=FLAGS.emitc_runtime_path,
  )
  model = tflite_to_cpp_lib.add_additional_headers(
      model=model, additional_headers=FLAGS.additional_headers)

  tflite_to_cpp_lib.print_llvm_registration_handle(
      model=model, base_class=FLAGS.base_class)

  model = tflite_to_cpp_lib.add_license_and_notice(model=model)

  if FLAGS.clang_format_path:
    model = tflite_to_cpp_lib.format_model(
        model=model,
        clang_format_path=FLAGS.clang_format_path,
        clang_format_style=FLAGS.clang_format_style,
    )

  cpp_path = tflite_to_cpp_lib.get_model_cpp_path(model, FLAGS.output_dir)
  hdr_path = tflite_to_cpp_lib.get_model_hdr_path(model, FLAGS.output_dir)

  logging.info('Writing generated files to [%s] and [%s].', cpp_path, hdr_path)
  with open(cpp_path, 'wt', encoding='utf-8') as f:
    f.write(model.cpp)
  with open(hdr_path, 'wt', encoding='utf-8') as f:
    f.write(model.hdr)
  logging.info('Done.')


if __name__ == '__main__':
  app.run(main)
