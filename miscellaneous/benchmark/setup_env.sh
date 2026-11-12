#!/bin/bash
# Build the benchmark conda env "sunrise-bench" from this checkout (like ci_basic.yml / sunrise_setup.sh),
# plus ffsim. Usage: bash miscellaneous/benchmark/setup_env.sh [conda_base]
set -ex
if [ -n "$1" ]; then source "$1/etc/profile.d/conda.sh"; else eval "$(conda shell.bash hook)"; fi
conda env remove -y -n sunrise-bench || true
conda create -y -n sunrise-bench python=3.11
conda activate sunrise-bench
cd "$(dirname "$0")/../.."
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install fqe --no-deps
pip install git+https://github.com/davibinco/TenCirChem-NG.git
pip install ffsim
pip install matplotlib
pip install -e .
pip freeze > miscellaneous/benchmark/env_$(hostname -s).txt
