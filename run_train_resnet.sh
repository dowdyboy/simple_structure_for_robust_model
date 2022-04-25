#!/usr/bin/env bash

PYTHONPATH="./":$PYTHONPATH \
python ./aisafe_glaring/aisafe_train_resnet.py $*
