#!/bin/bash

python3 -m venv _venv
source _venv/bin/activate
pip install -e torchhd
deactivate
