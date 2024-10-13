import os
import sys
from setuptools import setup, find_packages
from dfx import __version__


# required for building/installing from local sdist (.tar.gz) file
here = os.path.abspath(os.path.dirname(__file__))
os.chdir(here)


needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
tests_require = [
    'pytest',
    'pylint',
    'pytest-runner',
    'pytest-mock',
    'pytest-pylint',
    'mock'
]
pytest_runner = tests_require if needs_pytest else []

setup_requires = pytest_runner

with open('requirements.txt') as f:
    lines = (line.strip() for line in f.read().splitlines())
    lines = filter(lambda line: not line.startswith("#"), lines)
    install_requires = list(lines)

dev_require = list(set(tests_require))


setup(
    name="DFX",
    version=__version__,
    author="Aviv Nutovitz",
    author_email="avivnoto@gmail.com",
    description="",
    python_requires='>=3.8',
    setup_requires=setup_requires,
    tests_require=tests_require,
    # extras_require={'dev': dev_require},  # will install testing packages during dev
    install_requires=install_requires,
    packages=find_packages(exclude=['dfx_experiments']),
)
