import os

from setuptools import find_packages, setup


def read_requirements():
    with open("requirements.txt") as file:
        lines = file.readlines()
        requirements = [line.strip() for line in lines]
    return requirements


# get author and version information
VERSIONFILE = "src/sunrise/version.py"
info = {"__version__": None, "__author__": None}
with open(VERSIONFILE, "r") as f:
    for l in f:
        tmp = l.split("=")
        if tmp[0].strip().lower() in info:
            info[tmp[0].strip().lower()] = tmp[1].strip().strip('"').strip("'")

for k, v in info.items():
    if v is None:
        raise Exception(f"could not find {k} string in {VERSIONFILE}")  # noqa: TRY002

try:
    with open("README.md", "r") as f:
        long_description = f.read()
except Exception:  # noqa: BLE001
    long_description = ""

setup(
    name="project-sunrise",
    version=info["__version__"],
    author=info["__author__"],
    url="https://github.com/tequilahub/sunrise",
    description="Package for fermionic state preparation and simulation on quantum computers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=read_requirements(),
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "": [os.path.join("src")],
        "sunrise.CLPO.bin": ["*"],
    },
)
