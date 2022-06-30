## How to add a new feature

TL;DR; 3 steps:

- add the feature on the LLVM side
- tell training about the new feature
- retrain

A reference for the first two: LLVM [side](https://github.com/llvm/llvm-project/commit/99f00635d7acf1cbcdba35e7621f3a211aa3f237); and associated ml-compiler-opt [side](https://github.com/google/ml-compiler-opt/commit/882674933ce1c7a141591dfce0f2ae6e54a9fb9c)

## Adding the feature on the LLVM side

Most of the work here is choosing the feature and extracting it from IR, after
that, follow the existing pattern in `MLInlineAdvisor.cpp` (see `MLInlineAdvisor::getAdviceImpl`) or `MLRegallocEvictAdvisor.cpp` (see ` MLEvictAdvisor::extractFeatures`). Note that passing the feature to the ML model
happens generically, regardless how the model is evaluated (AOT or
development-mode); also, populating the training log happens generically, so you
do not need to worry about logging.

The key is to remember the name, type, and dimensions of the feature, they need
to match what we do in the next step.

## Training side

For each policy we train, there should be a `config.py`, e.g. `compiler_opt/rl/inlining/config.py`. Follow the example there.

## Retrain

First and foremost, **you must regenerate the vocabulary** - technically you
just need a vocab file for the new feature, but it's simpler to regenerate it
all. See the [demo section](demo/demo.md#collect-trace-and-generate-vocab)

**Note:** You only need to regenerate the vocabulary if the feature is going
to be normalized by a preprocessing layer for your model. If your feature does
not need to get put through a lambda normalization preprocessing layer, make sure
to regenerate the vocabulary and that your feature is added to the list
returned by `get_nonnormalized_features()` in `config.py`. In either case,
it is still quite simple and fast to just call the vocab generation again.

After that, retrain from [scratch](demo/demo.md#train-a-new-model).

## Notes

Currently, the LLVM side insists that all features it knows of be supported by a model. This means that we can't add a new feature, then use the previous trained policy as baseline for training. We are planning to relax this requirement and support using a previous policy as baseline - as long as the new feature set is
a superset of the old one.
