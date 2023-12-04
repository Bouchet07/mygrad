from setuptools import setup
from os import path
import io

here = path.abspath(path.dirname(__file__))

# get the dependencies and installs
with io.open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    all_reqs = f.read().split("\n")

setup(
    name='mygrad',
    version='0.1',
    description='Autograd',
    url='https://github.com/Bouchet07/mygrad',
    author='Diego Bouchet',
    license='MIT',
    packages=['mygrad'],
    install_requires=all_reqs
)
