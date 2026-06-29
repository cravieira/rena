#pragma once

#include <cstddef>

#include "hyle/common.hpp"
#include "hyle/bsc.hpp"

#include <hls_vector.h>

#ifndef __RN_FEATURES__
    #error "Define __RN_FEATURES__. Must be a value >0."
#endif
constexpr size_t RN_FEATURES = __RN_FEATURES__;

#ifndef __RN_CODEBOOK_SIZE__
    #error "Define __RN_CODEBOOK_SIZE__. Must be a value >0."
#endif
constexpr size_t RN_CODEBOOK_SIZE = __RN_CODEBOOK_SIZE__;

#ifndef __RN_DATAPATHS__
#define __RN_DATAPATHS__ 1
#endif
constexpr size_t RN_DATAPATHS = __RN_DATAPATHS__;



template <size_t N>
void hammdist_to_dotsim(hls::vector<int, N> &dot, const bsc_dist_t (&hamm)[N]) {
    // Converts hamming distance to dot similarity
    // Cast the result of bsc_distN to an hls::vector
    for (size_t i = 0; i < N; i++) {
        #pragma HLS unroll
        dot[i] = static_cast<int>(hamm[i]);
    }
    //dot = dot - __HV_DIMENSIONS__/2;
    //dot = __HV_DIMENSIONS__/2 - dot; // Map hamm_dist [0, D] to dot_sim [-D/2, D/2]
    dot = __HV_DIMENSIONS__ - 2*dot; // Map hamm_dist [0, D] to dot_sim [-D, D]
}

template <typename T, size_t N>
void hv_sign(bsc_hv_t &out, hls::vector<T, N> acc) {
    // Apply sign function to obtain binary HV
    for (size_t d = 0; d < __HV_SEGMENT_SIZE__; d++) {
        #pragma HLS unroll
        out[d] = acc[d] > 0;
    }
}

void unbind_module(bsc_hv_t &out, const bsc_hv_t &source, const bsc_hv_t (&features)[RN_FEATURES-1]);

void vector_projection(
    hls::vector<int, __HV_SEGMENT_SIZE__> &proj,
    const hls::vector<int, RN_CODEBOOK_SIZE> &dot_dists,
    const bsc_hv_t (&codebook)[RN_CODEBOOK_SIZE]
);

void rn_dp(
    bsc_hv_t (&pred)[HV_SEGMENTS],
    bool &converged,
    const bsc_hv_t (&input)[HV_SEGMENTS],
    const bsc_hv_t (&features)[HV_SEGMENTS][RN_FEATURES-1],
    const bsc_hv_t (&codebook)[HV_SEGMENTS][RN_CODEBOOK_SIZE],
    int convergence_threshold,
    int topaPT_threshold
    );

void rn(
    size_t (&pred_inds)[RN_FEATURES],
    size_t &iter,
    const bsc_hv_t (&input)[HV_SEGMENTS],
    const bsc_hv_t (&initial_guess)[RN_FEATURES][HV_SEGMENTS],
    const bsc_hv_t (&codebooks)[RN_FEATURES][HV_SEGMENTS][RN_CODEBOOK_SIZE],
    size_t max_iter,
    int convergence_threshold,
    int topaPT_threshold
    );
