#!/usr/bin/env bash

# Does a clean reinstall

python3 -m pip uninstall -qy master_thesis_mouritzen planetrl
rm -rf build dist planet/build planet/dist
python3 setup.py -q install
