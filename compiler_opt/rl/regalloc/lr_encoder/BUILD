# Inlining configurations.
load("//devtools/python/blaze:strict.bzl", "py_strict_library")
load("//devtools/python/blaze:pytype.bzl", "pytype_strict_binary", "pytype_strict_library")

licenses(["notice"])

package(
    default_applicable_licenses = ["//third_party/ml_compiler_opt:license"],
    default_visibility = [
        "//third_party/ml_compiler_opt:default_visibility",
    ],
)

filegroup(
    name = "gin_files",
    srcs = glob(["gin_configs/**"]),
)

py_strict_library(
    name = "config",
    srcs = ["config.py"],
    deps = [
        "//third_party/ml_compiler_opt/compiler_opt/rl:feature_ops",
        "//third_party/ml_compiler_opt/compiler_opt/rl/regalloc:config",
        "//third_party/py/gin",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
        "//third_party/py/tf_agents",
        "//third_party/py/tf_agents/specs:tensor_spec",
        "//third_party/py/tf_agents/trajectories:time_step",
    ],
)

pytype_strict_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        ":config",
        "//third_party/ml_compiler_opt/compiler_opt/rl:attention",
        "//third_party/py/gin",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
        "//third_party/py/tf_agents/utils:nest_utils",
    ],
)

pytype_strict_library(
    name = "dataset_ops",
    srcs = ["dataset_ops.py"],
    deps = [
        ":config",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
        "//third_party/py/tensorflow_text",
    ],
)

pytype_strict_library(
    name = "lr_encoder_runner",
    srcs = ["lr_encoder_runner.py"],
    deps = [
        "//third_party/ml_compiler_opt/compiler_opt/rl:compilation_runner",
        "//third_party/ml_compiler_opt/compiler_opt/rl:corpus",
        "//third_party/ml_compiler_opt/compiler_opt/rl:log_reader",
        "//third_party/py/gin",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

pytype_strict_library(
    name = "lr_encoder_problem_config",
    srcs = ["__init__.py"],
    deps = [
        ":config",
        ":lr_encoder_runner",
        "//third_party/ml_compiler_opt/compiler_opt/rl:problem_configuration",
        "//third_party/py/gin",
    ],
)

pytype_strict_binary(
    name = "train",
    srcs = ["train.py"],
    data = [":gin_files"],
    deps = [
        ":config",
        ":dataset_ops",
        ":model",
        "//third_party/ml_compiler_opt/compiler_opt/rl:policy_saver",
        "//third_party/ml_compiler_opt/compiler_opt/rl:registry",
        "//third_party/ml_compiler_opt/compiler_opt/rl/regalloc:config",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/logging",
        "//third_party/py/gin",
        "//third_party/py/tensorflow:tensorflow_google",  # build_cleaner: keep
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)
