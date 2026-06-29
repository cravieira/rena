#!/bin/bash

# Run several hdchog benchmarks to evaluate HLS accelerators for different HDC
# classes

set -eu

source bash/common.sh

JOBS=8 # Number of parallel jobs to be executed

VSAS="BSC"
DIMS="1536"
#DATAPATHS=$(seq 1 10)
DATAPATHS="1 2 4 6 8 10 12"
SEGMENT_SIZE="128"
CLOCK_PERIODS="5 10 15"

APP_NAME="" # Name of app to be executed
MAIN_TCL="" # Name of tcl script
PARAM_TEMP_DIR="_tmp_dse" # Temporary dir for parameter files

proj_name() {
    local dim=$2
    local ss=$3
    local dp=$4
    local cp=$5

    echo "vitis_dse_${APP_NAME}-d${dim}-seg_size${ss}-dp${dp}-cp${cp}"
}

dse() {
    mkdir -p $PARAM_TEMP_DIR
    cmd=""
    for vsa in $VSAS ; do
        for dim in $DIMS ; do
            for ss in $SEGMENT_SIZE; do
                for dp in $DATAPATHS ; do
                    for cp in $CLOCK_PERIODS ; do
                        p_name=$(proj_name $vsa $dim $ss $dp $cp)
                        cmd+=$(com_launch_synth "$MAIN_TCL" "$PARAM_TEMP_DIR" -dim $dim -seg-size "$ss" -vsa $vsa -datapath $dp -clk-period $cp -hls-synth "true" -vivado-impl "true" -project-name $p_name)
                    done
                done
            done
        done
    done
    echo "$cmd"
}

bsc_dse() {
    local dse_cmds=""

    # BSC
    VSAS="bsc"
    dse_cmds+=$(dse)

    echo "$dse_cmds"
}

APP_NAME="rn"
MAIN_TCL="${APP_NAME}.tcl"
cmd+=""
cmd+=$(bsc_dse)

printf "$cmd"
