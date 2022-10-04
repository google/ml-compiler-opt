
import gin

from compiler_opt.rl import problem_configuration
from compiler_opt.rl.regalloc import config
from compiler_opt.rl.regalloc_priority import regalloc_priority_runner


@gin.register(module='configs')
class RegallocPriorityConfig(problem_configuration.ProblemConfiguration):
    def get_runner(self, *args, **kwargs):
        return regalloc_priority_runner.RegAllocPriorityRunner(*args, **kwargs)

    def get_signature_spec(self):
        return config.get_regalloc_signature_spec()

    def get_preprocessing_layer_creator(self):
        return config.get_observation_processing_layer_creator()
