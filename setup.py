from __future__ import print_function
import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

try:
    import pandas
except ImportError:
    print('pandas is required during installation')
    sys.exit(1)

try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)

try:
    import sklearn
except ImportError:
    print('scikit-learn is required during installation')
    sys.exit(1)


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True
    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        import pytest
        pytest.main(self.test_args)


def readme():
    with open('README.rst.example') as f:
        return f.read()


setup(name='mcpt',
      version='0.0.1',
      description='A Fast Feature Evaluation method utilizing Monte Carlo Permutation Testing',
      #long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development'
      ],
      keywords='variable importance mutual information',
      url='http://github.com/dpatschke/mcpt',
      author='David Patschke',
      author_email='davidpatschke@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      test_suite='tests',
      include_package_data=True,
      zip_safe=False)
