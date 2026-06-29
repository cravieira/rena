#!/bin/bash

set -eu

DIM=1500
QUERIES=1024
SEED_START=0
SEED_STOP=1
BATCH_SIZE=512
BATCH_SIZE=256
M_START=4
M_STOP=8
CGR_BLOCK_SIZES="4 8 16 32 64"
#CGR_BLOCK_SIZES="4 8 16"
CGR_BLOCK_SIZES="4 8"
CGR_BUNDLE="mode opposite"
DECODING="sequential"

RESULTS_DIR="_results"

function main() {
    for seed in $(seq $SEED_START $SEED_STOP); do
        #python3 main.py --vsa MAP --M-log-start $M_START --M-log-stop $M_STOP --dim $DIM --decoding $DECODING --seed $seed --device cuda --codebooks 3 --queries $QUERIES --batchsize $BATCH_SIZE --csv-file ${RESULTS_DIR}/map-decoding_${DECODING}_seed${seed}.csv
        #python3 main.py --vsa FHRR --M-log-start $M_START --M-log-stop $M_STOP --dim $DIM --decoding $DECODING --seed $seed --device cuda --codebooks 3 --queries $QUERIES --batchsize 256 --csv-file ${RESULTS_DIR}/fhrr-decoding_${DECODING}_seed${seed}.csv
        #python3 main.py --vsa BSC --M-log-start $M_START --M-log-stop $M_STOP --dim $DIM --decoding $DECODING --seed $seed --device cuda --codebooks 3 --queries $QUERIES --batchsize 128 --csv-file ${RESULTS_DIR}/bsc-decoding_${DECODING}_seed${seed}.csv

        for cgr_block in $CGR_BLOCK_SIZES; do
            for cgr_bundle in $CGR_BUNDLE; do
                local div_batch=$(( $BATCH_SIZE/4 ))
                local batchsize=$(( $div_batch ? $div_batch : 1 ))
                python3 main.py --vsa CGR --M-log-start $M_START --M-log-stop $M_STOP --dim $DIM --decoding $DECODING --seed $seed --device cuda --codebooks 3 --queries $QUERIES --batchsize ${batchsize} --cgr-block-size ${cgr_block} --cgr-bundle $cgr_bundle --csv-file ${RESULTS_DIR}/cgr${cgr_block}-cgr_bundle_${cgr_bundle}-decoding_${DECODING}_seed${seed}.csv
            done
        done
    done
}

main
