# Run unit-tests on RN operations
open_project vitis_test_rn -reset

# Resonator Network defines
set RN_DIM 10
set RN_FEATURES 3
set RN_CODEBOOK_SIZE 4
set RN_CFLAGS "-D__RN_FEATURES__=${RN_FEATURES} -D__RN_CODEBOOK_SIZE__=${RN_CODEBOOK_SIZE}"

# Import Hyle
set HYLE_DIMENSIONS ${RN_DIM}
set HYLE_SEGMENT_SIZE ${HYLE_DIMENSIONS}
set HYLE_VSA "BSC"
source ./hyle/hyle.tcl

set cflags "${RN_CFLAGS} ${HYLE_CFLAGS}"

# Resonator Network sources
add_files -tb -cflags ${cflags} "src/test_rn.cpp"
add_files -cflags ${cflags} "src/rn.cpp"

set_top hello; # Set any top function
open_solution "solution1"
set_part  {xc7z020clg400-1}
create_clock -period 10
csim_design
