import os

from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt') as file:
        lines=file.readlines()
        requirements = [line.strip() for line in lines]
    return requirements

# get author and version information
VERSIONFILE = "src/sunrise/version.py"
info = {"__version__": None, "__author__": None}
with open(VERSIONFILE, "r") as f:
    for l in f.readlines():
        tmp = l.split("=")
        if tmp[0].strip().lower() in info:
            info[tmp[0].strip().lower()] = tmp[1].strip().strip('"').strip("'")

for k, v in info.items():
    if v is None:
        raise Exception("could not find {} string in {}".format(k, VERSIONFILE))

try:
    with open("README.md", "r") as f:
        long_description = f.read()
except Exception:
    long_description = ""

setup(
    name='project-sunrise',
    version="0.1.0",
    author="Chair of Quantum Algorithmics at Augsburg University (Germany)",
    url="https://github.com/tequilahub/sunrise",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=read_requirements(),
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    package_data={
        '': [os.path.join('src')]
    }
)
