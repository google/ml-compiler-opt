# How to Contribute

We'd love to accept your patches and contributions to this project. A good
starting step to get familiar with the project and set up a development
environment as per the inlining [demo](inlining-demo/demo.md). After running 
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
installed with `./versioned_pipenv sync --categories="dev-packages" --system`
and run on the entire repository with `yapf . -ir`.

## Linting

We use `pylint` to ensure that the code meets a certain set of linting
guidelines. To lint the repository, you can run the following command from
the root directory:

```
pylint --rcfile .pylintrc --recursive yes .
```

Pull requests will automatically be linted through Github Actions. You can find
the exact invocation used in the CI in `.github/workflows/main.yml`. We require
the lint job to pass before merging a PR.

## Typing

We use python type annotations to improve code quality. To validate our type
annotations, we use `pytype`. To run `pytype` against all the files in the
repository, you can run the following command:

```
pytype -j auto --overriding-parameter-count-checks .
```

Pull requests will automatically be type-checked through Github Actions. You
can find the exact invocation used in the CI in '.github/workflows/main.yml`.
We require the type-checking jobs to succeed before merging a PR.

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Community Guidelines

This project follows [Google's Open Source Community
Guidelines](https://opensource.google/conduct/).
