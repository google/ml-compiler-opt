---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
classes: wide
---

Welcome to MLGO --- Machine Learning Guided Compiler Optimizations!

MLGO is a framework for integrating ML techniques systematically in LLVM. It
replaces human-crafted optimization heuristics in LLVM with machine learned
models. MLGO currently supports two optimizations:

1.  [Inlining-for-Size](https://lists.llvm.org/pipermail/llvm-dev/2020-April/140763.html);
2.  [Register-Allocation-for-Performance](https://lists.llvm.org/pipermail/llvm-dev/2021-November/153639.html)

The compiler components are both available in the main LLVM repository.