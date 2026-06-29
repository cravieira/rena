#include "rn.hpp"
#include "prng.hpp"
#ifndef __SYNTHESIS__
#include "tracer.hpp"
#endif

#include <cstdint>
#include <hls_vector.h>

void unbind_module(bsc_hv_t &out, const bsc_hv_t &source, const bsc_hv_t (&features)[RN_FEATURES-1]) {
    // Create a batch of HVs to be processed by bsc_bindN()
    bsc_hv_t unbind_batch[RN_FEATURES];
    unbind_batch[0] = source;

    UnbindLoop:
    for (size_t i = 1; i < RN_FEATURES; i++) {
        unbind_batch[i] = features[i-1];
    }

    bsc_bindN<RN_FEATURES>(out, unbind_batch);
}

// A kernel for Vector-Matrix multiplication
void vector_projection(
        hls::vector<int, __HV_SEGMENT_SIZE__> &proj,
        const hls::vector<int, RN_CODEBOOK_SIZE> &dot_dists,
        const bsc_hv_t (&codebook)[RN_CODEBOOK_SIZE]
    ) {
    // Vector projection
    using sim_acc_t = hls::vector<int, __HV_SEGMENT_SIZE__>;
    sim_acc_t acc = 0;

    Accumulation:
    for (size_t m = 0; m < RN_CODEBOOK_SIZE; m++) {
        sim_acc_t scalar = dot_dists[m];
        sim_acc_t cb;

        // Convert 0s in codebooks to -1s
        BinaryToBipolar:
        for (size_t i = 0; i < __HV_SEGMENT_SIZE__; i++) {
            #pragma HLS unroll
            cb[i] = codebook[m][i] ? 1 : -1;
        }
        acc += scalar*cb;
    }

    proj = acc;
}


void rn_segment_sim_dp(
        const bsc_hv_t (&input),
        const bsc_hv_t (&features)[RN_FEATURES-1],
        const bsc_hv_t (&codebook)[RN_CODEBOOK_SIZE],
        size_t datapath_id,
        bsc_dist_t (&_dp_dist_acc)[RN_DATAPATHS][RN_CODEBOOK_SIZE]
    ) {
    // First stage of HV processing in RN computation

    bsc_hv_t unbound;
    unbind_module(unbound, input, features);

    #ifndef __SYNTHESIS__
    //tracer::write_unbind(unbound, input[s], features[s]); // TODO: Tracing needs fix to support segments
    #endif

    // Similarity search
    bsc_dist_t seg_dists[RN_CODEBOOK_SIZE];
    bsc_distN(seg_dists, unbound, codebook);

    // Parallel accumulation
    ParallelDistAccumulation:
    for (int m = 0; m < RN_CODEBOOK_SIZE; m++) {
        #pragma HLS unroll
        _dp_dist_acc[datapath_id][m] += seg_dists[m];
    }
}

uint32_t _prng_seed = 1231495;


void rn_dp(
    bsc_hv_t (&pred)[HV_SEGMENTS],
    bool &converged,
    const bsc_hv_t (&input)[HV_SEGMENTS],
    const bsc_hv_t (&features)[HV_SEGMENTS][RN_FEATURES-1],
    const bsc_hv_t (&codebook)[HV_SEGMENTS][RN_CODEBOOK_SIZE],
    int convergence_threshold,
    int topaPT_threshold
    ) {

    // Unbind features
    // Per datapath distance accumulator with a placeholder dimension for the
    // number of datapaths.
    static bsc_dist_t _dp_dist_acc[RN_DATAPATHS][RN_CODEBOOK_SIZE];

    parallel_reset(_dp_dist_acc);

    SegmentLoop:
    for (int s = 0; s < HV_SEGMENTS; s++) {
        rn_segment_sim_dp(
                input[s],
                features[s],
                codebook[s],
                s % RN_DATAPATHS,
                _dp_dist_acc
                );
    }

    // Accumulate distances computed by all datapaths
    bsc_dist_t dists[RN_CODEBOOK_SIZE];
    parallel_reset(dists);
    for (int dp = 0; dp < RN_DATAPATHS; dp++) {
        #pragma HLS unroll
        for (int f = 0; f < RN_CODEBOOK_SIZE; f++) {
            #pragma HLS unroll
            dists[f] += _dp_dist_acc[dp][f];
        }
    }

    // Convert hamming distance to dot similarity
    // TODO: Create a stricter data type for optimization
    hls::vector<int, RN_CODEBOOK_SIZE> dot_dists;
    hammdist_to_dotsim(dot_dists, dists);

    #ifndef __SYNTHESIS__
    //tracer::write_similarity(dot_dists);// TODO: Tracing needs fix to support segments
    #endif

    // Noise injection
    decltype(dot_dists) noise_vec;
    uint32_t final_state;
    rand_noise<RN_CODEBOOK_SIZE>(noise_vec, final_state, _prng_seed);
    _prng_seed = final_state;

    decltype(dot_dists) sim_noisy = dot_dists + noise_vec;

    // Activation function
    decltype(dot_dists) sim_act;
    ActivationLoop:
    for (int i = 0; i < RN_CODEBOOK_SIZE; i++) {
        sim_act[i] = sim_noisy[i] > topaPT_threshold ? sim_noisy[i] : 0;
    }

    // Convergence detection
    converged = false;
    ConvergenceThreshold:
    for (size_t i = 0; i < RN_CODEBOOK_SIZE; i++) {
        converged = converged | (sim_act[i] > convergence_threshold);
    }

    // Vector projection
    using sim_acc_t = hls::vector<int, __HV_SEGMENT_SIZE__>;
    SegmentProjectionLoop:
    for (int s = 0; s < HV_SEGMENTS; s++) {
        sim_acc_t acc;
        vector_projection(acc, sim_act, codebook[s]);
        // Apply sign function to obtain binary HV
        hv_sign(pred[s], acc);
    }
}

void rn(
    size_t (&pred_inds)[RN_FEATURES],
    size_t &iter,
    const bsc_hv_t (&input)[HV_SEGMENTS],
    const bsc_hv_t (&initial_guess)[RN_FEATURES][HV_SEGMENTS],
    const bsc_hv_t (&codebooks)[RN_FEATURES][HV_SEGMENTS][RN_CODEBOOK_SIZE],
    size_t max_iter,
    int convergence_threshold,
    int topaPT_threshold
    ) {
    static_assert(RN_FEATURES > 1, "Number of predicted FEATURES must be >= 2");

    bsc_hv_t pred_hvs[RN_FEATURES][HV_SEGMENTS];

    // TODO: Is this really necessary?
    // Read initial guess data into local buffer
    LoadInputSegment:
    for (int s = 0; s < HV_SEGMENTS; s++) {
        for (int f = 0; f < RN_FEATURES; f++) {
            #pragma HLS unroll
            pred_hvs[f][s] = initial_guess[f][s];
        }
    }

    bool converged = false;
    hls::vector<bool, RN_FEATURES> feat_converged;

    IterLoop:
    for (iter = 1; iter < max_iter; iter++) {
        #ifndef __SYNTHESIS__
        //tracer::set_iter(iter);
        #endif

        FeatLoop:
        for (int f = 0; f < RN_FEATURES; f++) {
            #ifndef __SYNTHESIS__
            //tracer::set_feature(f);
            #endif

            bsc_hv_t input_pred[HV_SEGMENTS][RN_FEATURES-1];
            // Pack input predictions for RN pass

            // This loop is just to make rn_dp call easier. It should not
            // impact performance as it is meant to be a mux switch among input
            // predictions.
            // UPDATE: Now implementing a small transposition to facilitate unbind_module implementation. Maybe a smarter unbind_module() in rn_dp() could avoid transposition.
            InputPacking:
            for (int s = 0; s < HV_SEGMENTS; s++) {
                int dst = 0;
                for (int src = 0; src < RN_FEATURES; src++) {
                    if (src != f) {
                        input_pred[s][dst] = pred_hvs[src][s];
                        dst++;
                    }
                }
            }

            // Execute RN pass
            rn_dp(pred_hvs[f], feat_converged[f], input, input_pred, codebooks[f], convergence_threshold, topaPT_threshold);
        }
        // Stop RN execution if any feature has converged
        if (feat_converged.reduce_or()) {
            break;
        }
    }

    // Return the most similar features after computing for max_iter iterations
    FinalPrediction:
    for (int f = 0; f < RN_FEATURES; f++) {
        // TODO: Encapsulate this snippet into a function
        bsc_dist_t dp_dists[1][RN_CODEBOOK_SIZE];

        parallel_reset(dp_dists[0]);
        for (int s = 0; s < HV_SEGMENTS; s++) {
            bsc_dist_t seg_dists[RN_CODEBOOK_SIZE];
            bsc_distN(seg_dists, pred_hvs[f][s], codebooks[f][s]);

            // Parallel accumulation
            ParallelDistAccumulation:
            for (int m = 0; m < RN_CODEBOOK_SIZE; m++) {
                #pragma HLS unroll
                dp_dists[0][m] += seg_dists[m];
            }
        }

        // Placeholder loop for to sum dist banks computed by datapaths
        //for (int dp = 0; dp < RN_DATAPATHS; dp++) {
        //
        //}
        bsc_dist_t (&dists)[RN_CODEBOOK_SIZE] = dp_dists[0];

        hls::vector<int, RN_CODEBOOK_SIZE> dot_dists;
        // TODO: Maybe it is possible to simplify the code by not converting
        // hammdist to dotsim and simply obtaining an special hammdist array
        // that can be argmin'ed
        hammdist_to_dotsim(dot_dists, dists);

        // Get absolute dot similarities
        AbsoluteDotSim:
        for (int i = 0; i < RN_CODEBOOK_SIZE; i++) {
            #pragma HLS unroll
            dot_dists[i] = dot_dists[i] < 0 ? -dot_dists[i] : dot_dists[i];
        }
        size_t argmax;
        parallel_argmax(argmax, dot_dists);
        pred_inds[f] = argmax;
    }
}
