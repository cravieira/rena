# Vitis-related helper functions
com_launch_vitis() {
    echo "vitis_hls $@\n"
}

com_parse_params() {
    tcl_content=""
    while [[ $# -gt 0 ]]; do
        case $1 in
            "-dim")
                tcl_content+="set RN_DIM $2\n"
                shift # past argument
                shift # past value
                ;;
            "-seg-size")
                tcl_content+="set RN_SEGMENT_SIZE $2\n"
                shift # past argument
                shift # past value
                ;;
            "-vsa")
                vsa_class=$2
                tcl_content+="set VSA $vsa_class\n"
                shift # past argument
                shift # past value
                ;;
            "-datapath")
                tcl_content+="set RN_DATAPATHS $2\n"
                shift # past argument
                shift # past value
                ;;
            "-clk-period")
                tcl_content+="set RN_CLK_PERIOD $2\n"
                shift # past argument
                shift # past value
                ;;
            "-hls-synth")
                tcl_content+="set RUN_HLS_SYNTH $2\n"
                shift # past argument
                shift # past value
                ;;
            "-vivado-synth")
                tcl_content+="set RUN_VIVADO_SYNTH $2\n"
                shift # past argument
                shift # past value
                ;;
            "-vivado-impl")
                tcl_content+="set RUN_VIVADO_IMPL $2\n"
                shift # past argument
                shift # past value
                ;;
            "-project-name")
                tcl_content+="set PROJECT_NAME $2\n"
                shift # past argument
                shift # past value
                ;;
            "*")
                echo "Unknown option $1"
                exit 1
                ;;
        esac
    done
    echo $tcl_content
}

com_launch_synth() {
    local tcl_script=$1
    shift
    local param_dir=$1
    shift

    param_file=$(mktemp "$param_dir/params.XXXX")
    file_content=$(com_parse_params "$@")
    printf "${file_content}" > $param_file
    echo $(com_launch_vitis $tcl_script $param_file)
}


# Launch a batch of jobs in parallel using GNU parallel.
# If any of the jobs fail, then all other jobs are immediately killed.
# $1: Number of simultaneous jobs
# $2: A string containing the commands to be launched separated by new line
function parallel_launch() {
    local jobs=$1
    local cmds="$2"

    # File to control the number of simultaneous jobs in gnu parallel. This bash
    # script launches a different number of jobs depending on the HDC scripts
    # running. This fine-grained control is useful to execute more jobs in
    # parallel in applications that are not so hardware demanding.
    PROCFILE=$(mktemp -p . _procfile.XXX)
    echo "Using \"$PROCFILE\" as procfile for job control in GNU parallel..."
    JOBLOG=$(mktemp -p . _joblog.XXX)
    echo "Using \"$JOBLOG\" as --joblog output for GNU parallel..."

    # Clean up temp file if script fails
    trap "rm $PROCFILE" QUIT TERM PWR EXIT

    echo "$jobs" > $PROCFILE
    #printf "$cmds" | parallel --verbose -j"$PROCFILE" --halt now,fail=1
    printf "$cmds" | parallel --verbose -j"$PROCFILE" --joblog "$JOBLOG"
}

