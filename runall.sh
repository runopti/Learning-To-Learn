#!/bin/sh

n_samplings=10
n_unroll=20
n_dimension=3
n_hidden=5
n_layers=2

max_epoch=20

python3 trainOptimizer.py --n_samplings $n_samplings --n_unroll $n_unroll --n_dimension $n_dimension --n_hidden $n_hidden --n_layers $n_layers 

python3 main.py --n_samplings $n_samplings --n_unroll $n_unroll --n_dimension $n_dimension --n_hidden $n_hidden --n_layers $n_layers --max_epoch $max_epoch
