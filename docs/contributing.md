# How to Contribute

We'd love to accept your patches and contributions to this project. A good
starting step to get familiar with the project and set up a development
enviroment as per the inlining [demo](inlining-demo/demo.md). After running 
through the demo, a good second step is to pick up an open
[issue](https://github.com/google/ml-compiler-opt/issues) or create one that you
would like to work on and submit a patch for. Please make sure that your patch
adheres to all the guidelines given below.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Code formatting

Use `yapf` to format the submission before making a PR. The version of yapf 
that is used by this repository along with other development tools can be 
installed with `pipenv sync --categories="dev-packages" --system` and run on 
the entire repository with `yapf . -ir`.

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Community Guidelines

This project follows [Google's Open Source Community
Guidelines](https://opensource.google/conduct/).
