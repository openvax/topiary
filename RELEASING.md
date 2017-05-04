# Releasing Topiary

This document explains what do once your [Pull Request](https://www.atlassian.com/git/tutorials/making-a-pull-request/) has been reviewed and all final changes applied. Now you're ready merge your branch into master and release it to the world:

1. Bump the [version](http://semver.org/) on __init__.py, as part of the PR you want to release.
2. Merge your branch into master.
3. After the Topiary unit tests complete successfully on Travis then the latest version
of the code (with the version specified above) will be pushed to [PyPI](https://pypi.python.org/pypi) automatically. If you're curious about how automatic deployment is achieved, see our [Travis configuration](https://github.com/hammerlab/topiary/blob/master/.travis.yml#L58).
