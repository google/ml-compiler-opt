import gin
import tensorflow as tf

import base64
import io
import os
import tempfile
from typing import Dict, Optional, Tuple
from absl import logging

from google.protobuf import struct_pb2
from compiler_opt.rl import compilation_runner


@gin.configurable(module='runners')
class RegAllocPriorityRunner(compilation_runner.CompilationRunner):
    def _compile_fn(
            self, file_paths: Tuple[str, ...], tf_policy_path: str, reward_only: bool,
            cancellation_manager: Optional[
                compilation_runner.WorkerCancellationManager]
    ) -> Dict[str, Tuple[tf.train.SequenceExample, float]]:

        file_paths = file_paths[0].replace('.bc', '')
        working_dir = tempfile.mkdtemp()

        log_path = os.path.join(working_dir, 'log')
        output_native_path = os.path.join(working_dir, 'native')

        result = {}
        try:
            command_line = []
            if self._launcher_path:
                command_line.append(self._launcher_path)
            command_line.extend([self._clang_path] + [
                                    '-c', file_paths, '-O3',
                                    '-mllvm', '-regalloc-priority-training-log=' + log_path,
                                    '-mllvm', '-regalloc-enable-priority-advisor=development',
                                    '-o', output_native_path
                                ])

            if tf_policy_path:
                command_line.extend(['-mllvm', '-regalloc-priority-model=' + tf_policy_path])
            compilation_runner.start_cancellable_process(command_line,
                                                         self._compilation_timeout,
                                                         cancellation_manager)

            sequence_example = struct_pb2.Struct()

            with io.open(log_path, 'rb') as f:
                sequence_example.ParseFromString(f.read())

            for key, value in sequence_example.fields.items():
                e = tf.train.SequenceExample()
                e.ParseFromString(base64.b64decode(value.string_value))
                print(e)
                if not e.HasField('feature_lists'):
                    continue
                r = (
                    e.feature_lists.feature_list['reward'].feature[-1].float_list
                        .value[0])
                if reward_only:
                    result[key] = (None, r)
                else:
                    del e.feature_lists.feature_list['reward']
                    result[key] = (e, r)

        finally:
            tf.io.gfile.rmtree(working_dir)

        return result
