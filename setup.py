from setuptools import setup, find_packages
import os

DESCRIPTION = 'Python to GPU compiler'

LONG_DESCRIPTION = None
try:
    LONG_DESCRIPTION = open('README.rst').read()
except:
    pass

setup(name='py2gpu',
      packages=find_packages(),
      author='Waldemar Kornewald',
      include_package_data=True,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      platforms=['any'],
      install_requires=[],
)
