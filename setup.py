import os

from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt') as file:
        lines=file.readlines()
        requirements = [line.strip() for line in lines]
    return requirements


setup(
    name='project-sunrise',
    version="0.1.0",
    author="Chair of Quantum Algorithmics at Augsburg University (Germany)",
    url="https://github.com/tequilahub/sunrise",
    install_requires=read_requirements(),
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    package_data={
        '': [os.path.join('src')]
    }
)
