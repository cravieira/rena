#!/usr/bin/python3

import argparse
import csv
from functools import partial
import json
import math
import numpy as np
from pathlib import Path
import torch
import torchhd
import torchhd.functional as functional
import sys

import src.rn as rn
import src.common as common
import src.hw_rand
import src.tracer as tracer
torch._logging.set_logs(graph_code=True)

class ExperimentExporter(object):
    """docstring for ExperimentExporter"""
    def __init__(self, path:str, max_iter, codebooks):
        super(ExperimentExporter).__init__()
        # TODO: Maybe check if path exists and if it is an empty directory
        self._path = path # Main path passed by the user
        self._max_iter = max_iter
        self._codebooks = codebooks
        self._input_inds = []
        self._scoreboard = []

    def register_batch(self, input_inds, scoreboard):
        '''
        Register a batch of RN samples.
        '''
        self._input_inds.append(input_inds)
        self._scoreboard.append(scoreboard)

    def save(self, experiment):
        input_inds = torch.vstack(self._input_inds).cpu().numpy()
        scoreboard = torch.vstack(self._scoreboard).cpu().numpy()
        Path(self._path).mkdir(parents=True)
        fmt = "%d"
        np.savetxt(f'{self._path}/queries.txt', input_inds, fmt=fmt)
        np.savetxt(f'{self._path}/scoreboard.txt', scoreboard, fmt=fmt)
        for i, cb in enumerate(self._codebooks):
            t = cb.cpu().numpy()
            np.savetxt(f'{self._path}/codebook{i}.txt', t, fmt=fmt)

        with open(f'{self._path}/experiment.json', 'w') as f:
            json.dump(experiment, f, indent=4)

def snr_to_std(d: int, snr: float):
    """
    Finds the std of a noise amplitude for a combination of SNR and hypervector dimensionality

    :param dimensions [int>0]: Dimensionality of the HVs
    :param snr [float]: Desired SNR in decibels (dB)
    """
    #std = 10**(math.log(d, 10) - snr/10)
    std = 10**(-snr/10)
    return std

def std_to_snr(d: int, std: float):
    """
    Finds the SNR of a combination of noise standard deviation and hypervector
    dimensionality. Returns the SNR in decibels.

    :param dimensions [int>0]: Dimensionality of the HVs
    :param std [float]: Standard deviation of the injected noise normalized by
    the dimensionality. Must be a value in the range [0, 1]
    """
    #snr = 10*math.log(d/std)
    snr = 10*math.log(1/std, 10)
    print(f'snr: {snr}')
    return snr

def float_range_type(min_val, max_val):
    """
    Returns a custom type function for argparse that validates a float within a given range.
    """
    def check_range(arg_value):
        try:
            f_value = float(arg_value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"'{arg_value}' is not a valid float.")

        if not (min_val <= f_value <= max_val):
            raise argparse.ArgumentTypeError(f"Value '{f_value}' not in range [{min_val}, {max_val}].")
        return f_value
    return check_range

def main():
    parser = argparse.ArgumentParser()
    vsa_options = {"MAP", "BSC", "FHRR", "CGR"}
    cgr_block_size = 4
    cgr_bundle = {"mode", "opposite"}
    cgr_bundle_default = "opposite"
    parser.add_argument("--vsa", choices=vsa_options, type=str, default="MAP")
    parser.add_argument("--cgr-block-size", type=int, default=cgr_block_size, help=f"Number of points in each dimension when using CGR. Defaults to {cgr_block_size}")
    parser.add_argument("--cgr-bundle", type=str, default=cgr_bundle_default, help=f"Bundling strategy when using CGR. Defaults to {cgr_bundle}")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim", type=int, default=1000)
    parser.add_argument("--batchsize", type=int, default=1, help="Batch size used in prediction. Defaults to 1.")
    parser.add_argument("--device", choices=["cpu", "cuda"], type=str, default="cpu", help=f"Choose the device used in experiments. Defaults to 'cpu'.")
    parser.add_argument("--max-iter", type=int, default=-1, help="Max number of iterations to run RNs. Defaults to a fraction of the search space.")
    parser.add_argument("--codebooks", type=int, default=3)
    parser.add_argument("--codebook-size", type=int, default=None)
    parser.add_argument("--M-log-start", type=int, default=None, help="Used in combination with --M-log-stop. Defines a logarithmic range to perform operational capacity experiments.")
    parser.add_argument("--M-log-stop", type=int, default=None, help="Used in combination with --M-log-start. Defines a logarithmic range to perform operational capacity experiments.")
    parser.add_argument("--nDecade", type=int, default=3, help="Used in combination with --M-log-start and --M-log-stop. Defines the number of points explored per decade of operational capacity.")
    parser.add_argument("--decoding", choices=["sequential", "parallel"], type=str, default="parallel", help=f"Choose the RN decoding. Defaults to 'parallel'.")
    parser.add_argument("--convergence-threshold", type=float_range_type(-1.0, 1.0), default=0.625, help=f"Threshold used in convergence detection. Must be a number in the range [-1, 1]. Defaults to '0.625'.")

    act_choices = rn.ACTIVATION_FUNCTIONS.keys()
    act_default = "identity"
    parser.add_argument("--activation", choices=act_choices, type=str, default=act_default, help=f"Choose the activation function. Default: {act_default}")
    #parser.add_argument("--topa", default=0, type=int, help=f"Choose the number of top attention values activated.")
    parser.add_argument("--topa", default=0, type=float, help=f"Choose the number of top attention values activated.")

    noise_choices = rn.NOISE_FUNCTIONS.keys()
    noise_default = "identity"
    parser.add_argument("--noise", choices=noise_choices, type=str, default=noise_default, help=f"Choose the noise injection function. Default: {noise_default}")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--normal-std", type=float_range_type(0.0, math.inf), default=None, help=f"Choose the std deviation in normal noise injection.")
    group.add_argument("--normal-snr", type=float_range_type(-math.inf, math.inf), default=None, help=f"Choose the normal noise amplitude in noise injection. The value passed must be in decibels (dB). This option sets the normal std based on SNR and must be mutually exclusive with --normal-std.")

    # Xorshift noise injection
    parser.add_argument("--xorshift-max", type=int, default=None, help=f"Select the max values in a xorshift noise injection. The same passed value is set for min values in xorshift distribution. Default to None.")
    default_xorshift_seed = 120321
    parser.add_argument("--xorshift-seed", type=int, default=default_xorshift_seed, help=f"Select the seed used in xorshift generation. Must be a value > 0. Default: {default_xorshift_seed}.")

    parser.add_argument("--queries", type=int, default=1, help="Number of random queries executed.")
    parser.add_argument("--csv-file", type=str, default=None, help="Write results in a csv file. The path is created if non-existent.")
    parser.add_argument("--export-exp", type=str, default=None, help="Export experiment to the given path. The path is created if it doesn't exist. Exported experiments contain information about the experiment ran and the codebooks used. When using with parameter --M-log-start and --M-log-stop, then each ran experiment is saved in a subfolder in the path assigned with --export-exp.")

    parser.add_argument("--tracer", type=str, default=None, help="Trace RN execution by writing computed tensors to files. The user should specify where the trace files should be created. Non-existent paths are created.. If tracing is active, then the batch size is reduced to 1.")
    args = parser.parse_args()

    vsa = args.vsa
    print(f"Running {vsa} model...")

    device = args.device
    torch.set_default_device(device)
    print(f"Using {device} device...")

    common.set_random_seed(args.seed)

    DIM = args.dim
    factors = args.codebooks

    # Create list of codebook experiments evluated based on arguments
    codebook_sizes = []
    if args.codebook_size:
        codebook_sizes.append(args.codebook_size)
    elif args.M_log_start and args.M_log_stop:
        nDecade = args.nDecade # Number of search space points evaluated per logarithmic decade.
        npoints = (args.M_log_stop - args.M_log_start) * nDecade + 1
        codebook_sizes = np.round(np.logspace(args.M_log_start, args.M_log_stop, npoints) ** (1/factors)).astype(int).tolist()
    else:
        raise RuntimeError("Please provide the codebook size for the experiment either by passing --codebook-size or --M-log-start and --M-log-stop.")
    print(f"Codebook sizes evaluated: {codebook_sizes}")

    BATCH_SIZE = args.batchsize
    if args.tracer:
        BATCH_SIZE = 1
        tracer.init(args.tracer)

    batch_iters = args.queries // BATCH_SIZE

    results = []
    # Iterate over all codebook sizes requested to sweep a search space
    for codebook_size in codebook_sizes:
        # Set attention function used based on args
        f_act = rn.ACTIVATION_FUNCTIONS[args.activation]
        # Select the appropriate threshold based on top-a argument
        topaPT_threshold = None
        if args.activation == "topaPT" or args.activation == "topaPT-safe":
            topaPT_threshold = rn.calc_topaPT_threshold(args.topa, args.dim, codebook_size)
            print(f"Codebook size: {codebook_size}, {args.activation} threshold: {topaPT_threshold}")
            f_act = partial(f_act, threshold=topaPT_threshold)

        # Parse noise injection parameters
        f_noise = rn.NOISE_FUNCTIONS[args.noise]
        normal_std = args.normal_std
        xs_std = None
        signal_noise_ratio = None
        if args.noise == "normal":
            if normal_std is not None:
                normal_std = normal_std
                signal_noise_ratio = std_to_snr(DIM, normal_std)
            else:
                normal_std = snr_to_std(DIM, args.normal_snr)
                signal_noise_ratio = args.normal_snr
                print(f'Converting SNRdb -> Normal: {args.normal_snr} -> {normal_std}')
            f_noise = partial(f_noise, std=normal_std)

        elif args.noise == "xorshift":
            prng = src.hw_rand.Xorshift32(args.xorshift_seed)
            xs_mean, xs_std = src.hw_rand.find_prng_stats(prng, args.xorshift_max) # TODO: Unimplemented for Xorshift32 class
            f_noise = partial(f_noise, prng=prng, max=args.xorshift_max, dim=args.dim)
        elif args.noise == "parallel_np_xorshift":
            prng = src.hw_rand.ParallelNumpyXorshift32(args.xorshift_seed, codebook_size*BATCH_SIZE)
            xs_mean, xs_std = src.hw_rand.find_prng_stats(prng, args.xorshift_max)
            signal_noise_ratio = std_to_snr(DIM, xs_std/DIM)
            f_noise = partial(f_noise, prng=prng, max=args.xorshift_max, dim=args.dim)

        # Set max number of iterations
        max_iter = args.max_iter
        iter_fac = 1
        if max_iter == -1:
            max_iter = int((codebook_size**(factors - 1)) / factors * iter_fac)

        search_space = codebook_size**factors
        print(f"Codebook size: {codebook_size}, Search space: {search_space}, Max iterations: {max_iter}")

        # Create codebooks
        extra_args = {'block_size': args.cgr_block_size} if vsa == 'CGR' else {}
        codebooks = [torchhd.random(codebook_size, DIM, vsa=vsa, **extra_args) for i in range(factors)]
        codebooks = torch.stack(codebooks, dim=0) # A tensor with all codebooks

        estimates = [torchhd.multiset(cb) for cb in codebooks]
        estimates = torch.stack(estimates, dim=0)
        estimates = functional.normalize(estimates)
        estimates = estimates.unsqueeze(0).repeat(BATCH_SIZE, 1, 1) # estimates = [Batch, Features, Dim]

        # Are we exporting this experiment's data to disk?
        exp_exporter = None
        if args.export_exp:
            export_path = args.export_exp
            # If the script is running in sweep mode, then save experiment in a subfolder
            if args.M_log_start or args.M_log_stop:
                export_path = f"{export_path}/M{codebook_size}"
            exp_exporter = ExperimentExporter(args.export_exp, max_iter, codebooks)

        correct_frames = 0
        correct_factors = 0
        niter_batches = []
        for batch_itr in range(batch_iters):
            print(f"Launching batch {batch_itr+1} of {batch_iters}...")
            # Create the combined symbols
            s, inds_rand = common.make_random_queries(codebooks, BATCH_SIZE)

            tracer.set_query(batch_itr) # Tracer only works with batches of 1
            tracer.register_input({"inds": inds_rand, "vector": s})

            final_estimates, convergence_iter = rn.rn_top(
                    s,
                    estimates,
                    codebooks,
                    max_iter,
                    args,
                    vsa=vsa,
                    decoding=args.decoding,
                    convergence_threshold=args.convergence_threshold,
                    activation=f_act,
                    noise=f_noise,
                    device=args.device
                )
            cmp = final_estimates == inds_rand
            correct_factors += cmp.sum().item()
            correct_frames += cmp.all(-1).sum().item()

            niter_batches.append(convergence_iter)

            if exp_exporter:
                exp_exporter.register_batch(inds_rand, final_estimates)

        acc_frames = correct_frames/args.queries
        acc_factors = correct_factors/(factors*args.queries)

        # Convergence speed results
        niters = torch.vstack(niter_batches)
        niter_max = niters.max().int().item()
        niter_min = niters.min().int().item()
        niter_avg = niters.mean().item()
        niter_std = niters.std().item()
        niter_q1 = niters.quantile(0.25).item()
        niter_q2 = niters.quantile(0.50).item()
        niter_q3 = niters.quantile(0.75).item()

        rst = {
            # Experiment parameters
            "vsa": vsa,
            "factors": factors,
            "codebook_size": codebook_size,
            "dimensions": DIM,
            "search_space": search_space,
            "seed": args.seed,
            "decoding": args.decoding,
            "convergence_threshold": args.convergence_threshold,
            "activation": args.activation,
            "topa": args.topa,
            "topaPT_threshold": topaPT_threshold,
            "noise": args.noise,
            "signal_noise_ratio": signal_noise_ratio, # The value in this field should be None if there is no noise injection
            "normal_std": normal_std,
            "xorshift_seed": args.xorshift_seed,
            "xorshift_max": args.xorshift_max,
            "xorshift_std": xs_std,
            "batch_size": BATCH_SIZE,
            "queries": args.queries,
            "cgr_block_size": args.cgr_block_size,
            "cgr_bundle": args.cgr_bundle,
            # Results
            "acc_factors": acc_factors,
            "acc_frames": acc_frames,
            "niter_avg": niter_avg,
            "niter_std": niter_std,
            "niter_min": niter_min,
            "niter_q1":  niter_q1,
            "niter_q2":  niter_q2,
            "niter_q3":  niter_q3,
            "niter_max": niter_max,
        }
        results.append(rst)
        print(f"seed: {args.seed}, queries: {args.queries}, acc_frame: {acc_frames}, acc_factor: {acc_factors}, niter_avg: {niter_avg:.2f}, niter_std: {niter_std:.2f}, niter_min: {niter_min}, niter_q1: {niter_q1}, niter_q2: {niter_q2}, niter_q3: {niter_q3}, niter_max: {niter_max}")

        # Save experiment's data
        if exp_exporter:
            exp_exporter.save(rst)

    # Write experiment results to csv file
    if args.csv_file:
        Path(args.csv_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())

            writer.writeheader()
            for rst in results:
                writer.writerow(rst)


if __name__ == '__main__':
    main()
