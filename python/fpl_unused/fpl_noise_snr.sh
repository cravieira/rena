#!/bin/bash

# Experiment - Normal noise injection sweep using SNR #
# Search in a range of normal noise standard devitions to find the one that
# provides the best RN model. This script uses the SNR as control knob for the
# noise. Please check the SNR definition in the source code.

set -eu

source common.sh
source _venv/bin/activate

QUERIES=100
BATCH_SIZE=$QUERIES
BATCH_SIZE=$QUERIES
SEED_START=0
SEED_STOP=0 # Inclusive
M_START=6
M_STOP=8
#M_START=11
#M_STOP=11
NDECADE=2
DECODING="sequential"
NORMAL_SNR_F2="16 17 18 19 20" # SNR for F2D1000
#NORMAL_SNR_F3="18 19 20 21 22" # SNR for F3D1500
NORMAL_SNR_F4="20 21 22 23 24 25 26 27" # SNR for F4D2000

PARALLEL_JOBS=4

RESULTS_DIR="_results_noise_fpl"
#
# Extended experiments
function main() {
    local seed=0
    local topa="7"
    local activation="topaPT"

    source _venv/bin/activate

    # Experiments with noise #
    for seed in $(seq $SEED_START $SEED_STOP); do
        for noise in $NORMAL_SNR_F2; do
            F=2
            D=1000
            #python3 main.py --vsa MAP --M-log-start $M_START --M-log-stop $M_STOP --nDecade $NDECADE --dim $D --decoding $DECODING --seed $seed --device cuda --codebooks $F --queries $QUERIES --batchsize $BATCH_SIZE --activation ${activation} --topa ${topa} --noise normal --normal-snr $noise --csv-file ${RESULTS_DIR}/map-F${F}D${D}-M${M_START}M${M_STOP}-noise_normal_${noise}-seed${seed}.csv
        done
    done

    # Experiments with noise #
    for seed in $(seq $SEED_START $SEED_STOP); do
        for noise in $NORMAL_SNR_F3; do
            F=3
            D=1500
            #python3 main.py --vsa MAP --M-log-start $M_START --M-log-stop $M_STOP --nDecade $NDECADE --dim $D --decoding $DECODING --seed $seed --device cuda --codebooks $F --queries $QUERIES --batchsize $BATCH_SIZE --activation ${activation} --topa ${topa} --noise normal --normal-snr $noise --csv-file ${RESULTS_DIR}/map-F${F}D${D}-M${M_START}M${M_STOP}-noise_normal_${noise}-seed${seed}.csv
        done
    done

    # Experiments with noise #
    for seed in $(seq $SEED_START $SEED_STOP); do
        for noise in $NORMAL_SNR_F4; do
            F=4
            D=2000
            python3 main.py --vsa MAP --M-log-start $M_START --M-log-stop $M_STOP --nDecade $NDECADE --dim $D --decoding $DECODING --seed $seed --device cuda --codebooks $F --queries $QUERIES --batchsize $BATCH_SIZE --activation ${activation} --topa ${topa} --noise normal --normal-snr $noise --csv-file ${RESULTS_DIR}/map-F${F}D${D}-M${M_START}M${M_STOP}-noise_normal_${noise}-seed${seed}.csv
        done
    done
}

cmds=$(main)
parallel_launch
