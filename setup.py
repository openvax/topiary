# Copyright (c) 2017. Mount Sinai School of Medicine
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import os
import re

from setuptools import setup, find_packages

readme_dir = os.path.dirname(__file__)
readme_path = os.path.join(readme_dir, 'README.md')

try:
    with open(readme_path, 'r') as f:
        readme_markdown = f.read()
except:
    readme_markdown = ""

try:
    import pypandoc
    readme_restructured = pypandoc.convert(readme_markdown, to='rst', format='md')

    # create README.rst for deploying on Travis
    rst_readme_filename = readme_path.replace(".md", ".rst")
    with open(rst_readme_filename, "w"):
        f.write(readme_restructured)
except:
    print(
        "Conversion of long_description from MD to reStructuredText failed...")

with open('topiary/__init__.py', 'r') as f:
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        f.read(),
        re.MULTILINE).group(1)

if not version:
    raise RuntimeError('Cannot find version information')

if __name__ == '__main__':
    setup(
        name='topiary',
        version=version,
        description="Predict cancer epitopes from cancer sequence data",
        author="Alex Rubinsteyn, Tavi Nathanson",
        author_email="alex.rubinsteyn@gmail.com",
        url="https://github.com/hammerlab/topiary",
        license="http://www.apache.org/licenses/LICENSE-2.0.html",
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Environment :: Console',
            'Operating System :: OS Independent',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
        ],
        install_requires=[
            'numpy >=1.7, <2.0',
            'pandas >=0.13.1',
            'mhctools >= 1.3.0',
            'varcode >=0.3.17',
            'nose >=1.3.6',
            'gtfparse >=0.0.4',
            'mhcnames',
        ],
        long_description=readme_restructured,
        packages=find_packages(exclude="test"),
        entry_points={
            'console_scripts': [
                'topiary = topiary.cli.script:main'
            ]
        }
    )
