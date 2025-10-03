#!/bin/bash

source dj-conda-py3.8/bin/activate

python process_kernels.py
python generate_rdf.py
