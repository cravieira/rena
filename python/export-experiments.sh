#!/bin/bash

# Export experiments that can be loaded into HLS environment.

source _venv/bin/activate

DEVICE="cuda"
EXPS_PATH="_experiments"
BATCHSIZE=512
queries=1024
features=3
dim=256
M=10
#python3 main.py --queries $queries --dim $dim --vsa BSC --codebooks $features --codebook-size $M --decoding sequential --max-iter -1 --device $DEVICE --batchsize $BATCHSIZE --export-exp "${EXPS_PATH}/_d${dim}_f${features}_m${M}"
#
#dim=1500
#M=100
#python3 main.py --queries $queries --dim $dim --vsa BSC --codebooks $features --codebook-size $M --decoding sequential --max-iter -1 --device $DEVICE --batchsize $BATCHSIZE --export-exp "${EXPS_PATH}/_d${dim}_f${features}_m${M}"
#
#dim=1536
#M=100
#python3 main.py --queries $queries --dim $dim --vsa BSC --codebooks $features --codebook-size $M --decoding sequential --max-iter -1 --device $DEVICE --batchsize $BATCHSIZE --export-exp "${EXPS_PATH}/_d${dim}_f${features}_m${M}"


dim=256
M=15
python3 main.py --queries $queries --dim $dim --vsa BSC --codebooks $features --codebook-size $M --decoding sequential --max-iter -1 --device $DEVICE --batchsize $BATCHSIZE --export-exp "${EXPS_PATH}/_d${dim}_f${features}_m${M}"

