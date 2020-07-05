#!/bin/bash
conda env create -f environment.yml
conda activate message-passing-nn
#export PYTHONPATH=path/to/message-passing-neural-network/conda remove --name myenv --all
. inference.sh
python message_passing_nn/cli.py inference