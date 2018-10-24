import sys
from setuptools import setup, find_packages

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release

# version.py defines the VERSION and VERSION_SHORT variables.
# We use exec here so we don't import allennlp whilst setting up.
VERSION = {}
with open("adversarialnlp/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

# make pytest-runner a conditional requirement,
# per: https://github.com/pytest-dev/pytest-runner#considerations
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

with open('requirements.txt', 'r') as f:
    install_requires = [l for l in f.readlines() if not l.startswith('# ')]

setup_requirements = [
    # add other setup requirements as necessary
] + pytest_runner

setup(name='adversarialnlp',
      version=VERSION["VERSION"],
      description='A generice library for crafting adversarial NLP examples, built on AllenNLP and PyTorch.',
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      classifiers=[
          'Intended Audience :: Science/Research',
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      keywords='adversarialnlp allennlp NLP deep learning machine reading',
      url='https://github.com/huggingface/adversarialnlp',
      author='Thomas WOLF',
      author_email='thomas@huggingface.co',
      license='Apache',
      packages=find_packages(exclude=["*.tests", "*.tests.*",
                                      "tests.*", "tests"]),
      install_requires=install_requires,
      scripts=["bin/adversarialnlp"],
      setup_requires=setup_requirements,
      tests_require=[
          'pytest',
      ],
      include_package_data=True,
      python_requires='>=3.6.1',
      zip_safe=False)
