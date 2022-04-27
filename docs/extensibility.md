## Extending to new optimization problems

This guide is about extending the training tools to support new optimization
problems. It is assumed the necessary LLVM changes have been made - i.e.
instrumenting the optimization pass with a way to carry out decision making via
a trained model, training log collection - see the
lib/Analysis/MLInlineAdvisor.cpp and lib/CodeGen/MLRegallocEvictAdvisor.cpp for
examples.


### Extensibility steps

Refer to `compiler_opt/rl/inlining` or `compiler_opt/rl/regalloc`.

1) create a directory peer to `inlining` and `regalloc`. This placement is
not necessary, but sufficient for illustration.

2) define the implementation of
`compiler_opt.rl.compilation_runner.CompilationRunner` that's specific to your
problem. Refer to the examples. Note how we always start processes via the
`compiler_opt.rl.start_cancellable_process()` utility.

3) define the ML interface - see the `config.py` file in each of the examples.

4) extend `compiler_opt.rl.problem_configuration.ProblemConfiguration`. Make the
new class gin-configurable. By convention, define this in the `__init__.py`.

5) place specific gin configs in the subdirectory, as well as vocab (these are
optional, but likely necessary). A convention here is to make sure your gin
files make the configurable `config_registry.get_configuration.implementation`
point to your implementation of `ProblemConfiguration`. See the `common.gin`
files in our examples. This allows any tool to just pick up your problem when
pointing it (via `--gin_files`) to your problem.

You can have multiple gin files for different algorithm configurations, and
reuse common settings (like the above) via gin's `import` mechanism. See our
examples where we have different configs for PPO or behavioral cloning.

6) add your module to the list in `compiler_opt.rl.registry.py`, under the
"Register implementations" comment.

 'compilation problem' is an optimization problem with a specific way of
invoking clang and specific features and tensorflow topologies. The component
model requires all these be exported in a class implementing
ProblemConfiguration below, however, to avoid cycle dependencies in Bazel
environments, do not explicitly inherit from it.

Internally, all the module's implementation parameters are expected to be
gin-initialized.

### Use

Existing tools (e.g. `train_locally.py`) will just transparently use your new
component if you point the tool to one of your gin files. This assumes your gin
file binds `config_registry.get_configuration.implementation` as described:

`--gin_bindings=config_registry.get_configuration.implementation=@configs.InliningConfig`

To use in a new tool:

*   just get a ProblemConfiguration object in your python:

    `config = problem_configuration.get_configuration()`

*   make sure your tool also exposes `--gin_files` and `--gin_bindings` and
    bootstraps gin.

### Conventions

* to avoid long binding names, use the `runners` module name for the
  `CompilationRunner` implementation, and use the `configs` module name for the
  implementation of `ProblemConfiguration`.

* the `CompilationRunner` gin initialization should initialize to None, and use,
  the `clang_path` and `launcher_path` macros
  (https://github.com/google/gin-config#syntax-quick-reference):

```
  clang_path = None
  launcher_path = None
  runners.MyCompilationRunner.clang_path = %clang_path
  runners.MyCompilationRunner.launcher_path = %launcher_path
```

Use a similar pattern for problem-specific additional flags (see inlining's
`llvm_size_path` for example). When running tools, this allows the user pass
common flags transparently wrt the underlying runner - i.e. if swapping 2
runners, the clang flag stays the same:
`--gin_bindings=clang_path="'/foo/bar/clang'"`
