#!/bin/bash
conda env create -f environment.yml
conda activate message-passing-neural-network
#export PYTHONPATH=path/to/message-passing-neural-network/
export PARAMETERS_PATH=./parameters/inference.json
python message_passing_nn/cli.py inference