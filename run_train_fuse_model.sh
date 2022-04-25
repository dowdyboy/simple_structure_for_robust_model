#!/usr/bin/env bash

PYTHONPATH="./":$PYTHONPATH \
python ./aisafe_glaring/aisafe_train_fuse_model.py $*
