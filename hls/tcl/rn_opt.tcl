#Tests for D256 M 21 - top: rn_dp()
# IO Directives
set_directive_interface -mode ap_memory -storage_type rom_1p rn_dp features; # Should be removed for rn()
set_directive_interface -mode ap_memory -storage_type rom_1p rn_dp codebook; # Should be changed to "codebooks" for rn()

# rn_dp IO
set_directive_array_partition -dim 1 -type cyclic -factor ${RN_DATAPATHS} rn_dp input
set_directive_array_partition -dim 1 -type cyclic -factor ${RN_DATAPATHS} rn_dp codebook
set_directive_array_partition -dim 1 -type cyclic -factor ${RN_DATAPATHS} rn_dp features

# Disable pipelines on Segment Loops to avoid II Violations and Timing issues
set_directive_pipeline -off rn_dp/SegmentLoop
set_directive_pipeline -off rn_dp/SegmentProjectionLoop

set_directive_inline vector_projection; # Greatly improves design
set_directive_inline unbind_module
set_directive_inline bsc_distN
set_directive_inline bsc_dist
set_directive_unroll bsc_dist/AddReduce

# Unroll activation, convergence detection
set_directive_unroll rn_dp/ActivationLoop;
set_directive_unroll rn_dp/ConvergenceThreshold;

# Segment datapath directives
# Useless directives
#set_directive_unroll rn_dp/SegmentLoop -factor ${RN_DATAPATHS}
#set_directive_unroll rn_dp/SegmentProjectionLoop -factor ${RN_DATAPATHS}
#

set_directive_unroll -factor ${RN_DATAPATHS} rn_dp/SegmentLoop
set_directive_array_partition rn_dp _dp_dist_acc
# Explicitely instantiate datapath
set_directive_function_instantiate rn_segment_sim_dp datapath_id
set_directive_unroll -factor ${RN_DATAPATHS} rn_dp/SegmentProjectionLoop


# Using only the AddReduce unroll greatly reduces the latency. Now it is necessary to fix the input data
# Optimize distance computation
#set_directive_unroll bsc_distN/ComputeDist; # Unroll parallel dimensions accumulation (vertical)
set_directive_unroll bsc_dist/AddReduce; # Unroll parallel dimensions accumulation (vertical)
