#!/bin/bash

# Run baseline experiments for F3D1500 baseline RN #
# This script makes experiments considering a baseline RN and a baseline RN
# with optimizations but without noise injection.

set -eu

source common.sh
source _venv/bin/activate

F=3
DIM=1500
QUERIES=100
BATCH_SIZE=$QUERIES
SEED_START=0
SEED_STOP=4 # Inclusive
M_START=6
NDECADE=3
DECODING="sequential"

PARALLEL_JOBS=1

RESULTS_DIR="_results_baseline_fpl"

function main() {
    for seed in $(seq $SEED_START $SEED_STOP); do
        # Fails for 10^8
        echo "python3 main.py --vsa MAP --nDecade $NDECADE --M-log-start $M_START --M-log-stop 8 --dim $DIM --decoding $DECODING --seed $seed --device cuda --codebooks 3 --queries $QUERIES --batchsize $BATCH_SIZE --csv-file ${RESULTS_DIR}/baseline_F${F}D${DIM}_seed${seed}.csv"
    done

}

cmds=""
cmds+=$(main)
parallel_launch $PARALLEL_JOBS "$cmds"
#echo "${cmds}"
