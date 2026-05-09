Contributing
============

Releases
--------

The release workflow in `./github/workflows/publish.yaml` is triggered upon pushing a commit tagged with a valid [semantic version](https://semver.org) to main. Release versions are published on PyPI, pre-release versions on TestPyPI. To get started, it is recommended to first publish a pre-release version on TestPyPI, which will not affect the official version on PyPI. Don't forget to also bump the version in `./pyproject.toml`.
