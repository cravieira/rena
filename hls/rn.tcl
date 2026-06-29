# RN testbench
# RN accelerator parameters
set RN_DIM 256
set RN_DIM 1536
set RN_SEGMENT_SIZE 128
set RN_FEATURES 3
set RN_CODEBOOK_SIZE 100
set RN_DATAPATHS 12
set RN_CLK_PERIOD 10

# General project variables
set RUN_SIM false
set RUN_HLS_SYNTH false
set RUN_VIVADO_SYNTH false
set RUN_VIVADO_IMPL false
set PROJECT_NAME ""

set RUN_VIVADO_SYNTH true
set RUN_VIVADO_IMPL true

# Is there a custom parameter file?
if {$argc == 1} {
    source [lindex $argv 0]
}

set script_path [ file dirname [ file normalize [ info script ] ] ]; # Path of this script file

set proj_name "vitis_rn_d${RN_DIM}-ss${RN_SEGMENT_SIZE}-f${RN_FEATURES}_m${RN_CODEBOOK_SIZE}_dp${RN_DATAPATHS}"
if { $PROJECT_NAME != "" } {
    set proj_name ${PROJECT_NAME}
}

open_project "${proj_name}" -reset
#open_project "${proj_name}"

# Resonator Network defines
set RN_CFLAGS "-D__RN_FEATURES__=${RN_FEATURES} -D__RN_CODEBOOK_SIZE__=${RN_CODEBOOK_SIZE} -D__RN_DATAPATHS__=${RN_DATAPATHS}"

# Define testbench constants
set HYLE_DIMENSIONS ${RN_DIM}
set HYLE_SEGMENT_SIZE ${RN_SEGMENT_SIZE}
set HYLE_VSA "BSC"
source ./hyle/hyle.tcl

set cflags "${RN_CFLAGS} ${HYLE_CFLAGS}"

# Resonator Network sources
add_files -tb -cflags ${cflags} "src/rn_tb.cpp"
add_files -tb -cflags "${cflags} -std=c++17 -lstdc++fs" "src/tracer.cpp"
add_files -cflags ${cflags} "src/rn.cpp"
add_files -cflags ${cflags} "src/prng.cpp"

set_top rn_dp;
open_solution "solution1"
#set_part  {xc7z020clg400-1}; # Zynq board
#set_part {xczu59dr-ffvf1760-2-i}; # Zynq UltraScale+
#set_part xc7a200tfbg484-1; # Artix7 UltraScale+
set_part xczu7cg-fbvb900-1-e; # Zynq UltraScale+
create_clock -period ${RN_CLK_PERIOD}

if {bool($RUN_SIM)} {
    csim_design -argv "${script_path}/_experiments/q100_d${RN_DIM}_f${RN_FEATURES}_m${RN_CODEBOOK_SIZE}"
    #csim_design  -sanitize_address -sanitize_undefined -argv "${script_path}/_experiments/q100_d${RN_DIM}_f${RN_FEATURES}_m${RN_CODEBOOK_SIZE}"
}

source tcl/rn_opt.tcl
csynth_design

if {bool($RUN_HLS_SYNTH)} {
    csynth_design
}

if {bool($RUN_VIVADO_SYNTH)} {
    export_design -flow syn
}
if {bool($RUN_VIVADO_IMPL)} {
    export_design -flow impl
}
