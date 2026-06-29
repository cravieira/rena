#!/bin/bash

# Experiment - PRNG noise injection sweep with xorshift #
# Search different PRNG configurations by varying the max values produced with
# xorshift.

set -eu

source common.sh
source _venv/bin/activate

DIM=1500
QUERIES=100
BATCH_SIZE=$QUERIES
SEED_START=0
SEED_STOP=4 # Inclusive
M_START=6
M_STOP=11
NDECADE=1 # Number of points evaluated between search space decades
DECODING="sequential"
NOISE="parallel_np_xorshift"
XORSHIFT_MAX="2 4 8 16 32" # XS 32 is already a bad choice. No need to explore further

ACTIVATION="topaPT"
TOPA="14.7"

PARALLEL_JOBS=4

RESULTS_DIR="_results_noise_prng_fpl"

function main() {
    mkdir -p $RESULTS_DIR

    for seed in $(seq $SEED_START $SEED_STOP); do
        for xs_max in $XORSHIFT_MAX ; do
            echo "python3 main.py --vsa MAP --M-log-start $M_START --M-log-stop $M_STOP --nDecade $NDECADE --dim $DIM --decoding $DECODING --seed $seed --device cuda --codebooks 3 --queries $QUERIES --batchsize $BATCH_SIZE --activation ${ACTIVATION} --topa ${TOPA} --noise $NOISE --xorshift-max $xs_max --csv-file ${RESULTS_DIR}/map-M${M_START}M${M_STOP}-${ACTIVATION}_${TOPA}-noise_${NOISE}-xsmax_${xs_max}-seed${seed}.csv"
        done
    done
}

cmds=""
cmds+=$(main)
parallel_launch $PARALLEL_JOBS "$cmds"
#echo "$cmds"
