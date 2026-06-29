#!/bin/bash

# Experiment - Normal noise injection sweep #
# Search in a range of normal noise standard devitions to find the one that
# provides the best RN model.

set -eu

source common.sh
source _venv/bin/activate

QUERIES=100
BATCH_SIZE=$QUERIES
SEED_START=0
SEED_STOP=4 # Inclusive
M_START=6
M_STOP=11
NDECADE=1
DECODING="sequential"
NOISE="0.005 0.007 0.009 0.011 0.013"
NORMAL_STD_F2="0.005 0.007 0.009 0.011 0.013"
NORMAL_STD_F3="0.005 0.007 0.009 0.011"
NORMAL_STD_F4="0.005 0.007 0.009 0.011 0.013"

PARALLEL_JOBS=5

RESULTS_DIR="_results_noise_std_fpl"
#
# Extended experiments
function main() {
    local seed=0
    local topa="14.7"
    local activation="topaPT"

    source _venv/bin/activate

    # Experiments with noise #
    for seed in $(seq $SEED_START $SEED_STOP); do
        for noise in $NORMAL_STD_F2; do
            F=2
            D=1000
            #python3 main.py --vsa MAP --M-log-start $M_START --M-log-stop $M_STOP --nDecade $NDECADE --dim $D --decoding $DECODING --seed $seed --device cuda --codebooks $F --queries $QUERIES --batchsize $BATCH_SIZE --activation ${activation} --topa ${topa} --noise normal --normal-snr $noise --csv-file ${RESULTS_DIR}/map-F${F}D${D}-M${M_START}M${M_STOP}-noise_normal_${noise}-seed${seed}.csv
        done
    done

    # Experiments with noise #
    for seed in $(seq $SEED_START $SEED_STOP); do
        for noise in $NORMAL_STD_F3; do
            F=3
            D=1500
            echo "python3 main.py --vsa MAP --M-log-start $M_START --M-log-stop $M_STOP --nDecade $NDECADE --dim $D --decoding $DECODING --seed $seed --device cuda --codebooks $F --queries $QUERIES --batchsize $BATCH_SIZE --activation ${activation} --topa ${topa} --noise normal --normal-std $noise --csv-file ${RESULTS_DIR}/map-F${F}D${D}-M${M_START}M${M_STOP}-noise_normal_${noise}-seed${seed}.csv"
        done
    done

    ## Experiments with noise #
    #for seed in $(seq $SEED_START $SEED_STOP); do
    #    for noise in $NORMAL_STD_F4; do
    #        F=4
    #        D=2000
    #        python3 main.py --vsa MAP --M-log-start $M_START --M-log-stop $M_STOP --nDecade $NDECADE --dim $D --decoding $DECODING --seed $seed --device cuda --codebooks $F --queries $QUERIES --batchsize $BATCH_SIZE --activation ${activation} --topa ${topa} --noise normal --normal-snr $noise --csv-file ${RESULTS_DIR}/map-F${F}D${D}-M${M_START}M${M_STOP}-noise_normal_${noise}-seed${seed}.csv
    #    done
    #done
}

cmds=""
cmds+=$(main)
parallel_launch $PARALLEL_JOBS "$cmds"
