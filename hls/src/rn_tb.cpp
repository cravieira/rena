#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <string>
#include <numeric>

#include "hyle/defines.hpp"
#include "hyle/bsc.hpp"
#include "hyle/dataset.hpp"
#include "rn.hpp"
#include "tracer.hpp"

template <size_t F, size_t  S, size_t  M>
void transpose(bsc_hv_t (&out)[S][F][M], const bsc_hv_t (&in)[F][S][M]) {
        for (size_t f = 0; f < F; f++) {
            for (size_t s = 0; s < S; s++) {
                for (size_t m = 0; m < M; m++) {
                out[s][f][m] = in[f][s][m];
            }
        }
    }
}

void parse_codebooks(bsc_hv_t (&codebooks)[RN_FEATURES][HV_SEGMENTS][RN_CODEBOOK_SIZE], const std::string &path) {
    using cb_t = bsc_hv_t[HV_SEGMENTS][RN_CODEBOOK_SIZE];
    //cb_t cbs[RN_FEATURES];
    for (int i = 0; i < RN_FEATURES; i++) {
        auto i_str = std::to_string(i);
        auto cb_path = path+"/codebook"+i_str+".txt";
        auto cb_data = parse_mem_file(cb_path);
        //segment_mem<RN_CODEBOOK_SIZE>(cb_data, cbs[i]);

        segment_mem<RN_CODEBOOK_SIZE>(cb_data, codebooks[i]);
    }

    //transpose<RN_FEATURES, HV_SEGMENTS, RN_CODEBOOK_SIZE>(codebooks, cbs);
}

float mean(std::vector<float> vec) {
    float acc = 0;
    for (auto i : vec) {
        acc += i;
    }
    float elems = static_cast<float>(vec.size());
    float mean = static_cast<float>(acc)/elems;
    return mean;
}

// Based on stackoverflow
// https://stackoverflow.com/a/7616783
float stdev(std::vector<float> v) {
    float sum = std::accumulate(v.begin(), v.end(), 0.0);
    float mean = sum / v.size();

    std::vector<float> diff(v.size());
    std::transform(v.begin(), v.end(), diff.begin(), [mean](float x) { return x - mean; });
    float sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    float stdev = std::sqrt(sq_sum / v.size());
    return stdev;
}

int main(int argc, char *argv[]) {
    //tracer::init("./_tracer");

    auto exp_path = std::string(argv[1]);

    // Parse queries made in the experiment
    auto query_path = exp_path+"/queries.txt";
    auto queries = read_dataset(query_path);

    // Parse the answers of the python RN implementation for this experiment
    auto scoreboard_path = exp_path+"/scoreboard.txt";
    auto scoreboard = read_dataset(scoreboard_path);

    // Parse codebooks
    bsc_hv_t cbs[RN_FEATURES][HV_SEGMENTS][RN_CODEBOOK_SIZE];
    parse_codebooks(cbs, exp_path);

    bsc_hv_t initial_estimates[RN_FEATURES][HV_SEGMENTS];
    // Compute the initial estimation of all features
    for (int s = 0; s < HV_SEGMENTS; s++) {
        for (int f = 0; f < RN_FEATURES; f++) {
            bsc_bundleN(initial_estimates[f][s], cbs[f][s]);
        }
    }

    uint64_t correct_frames = 0;
    uint64_t correct_factors = 0;
    // TODO: Parse max_iter from the experiments file
    int iter_fac = 1;
    size_t max_iter = std::pow(RN_CODEBOOK_SIZE, (RN_FEATURES - 1)) / RN_FEATURES * iter_fac;
    // TODO: Parse convergence_threshold from the experiments file
    int convergence_threshold = std::round(DIM*0.625);
    // TODO: Parse topaPT_threshold from the experiments file
    int topaPT_threshold = std::round(DIM*0.0269);

    std::vector<float> niters;

    for (int i = 0; i < queries.size(); i++) {
        tracer::set_query(i);

        // Compute S vector from queries
        const auto &query_inds = queries[i];
        const bsc_hv_t *query_vectors[HV_SEGMENTS][RN_FEATURES];
        for (int f = 0; f < RN_FEATURES; f++) {
            for (int s = 0; s < HV_SEGMENTS; s++) {
                auto ind = query_inds[f];
                query_vectors[s][f] = &cbs[f][s][ind];
            }
        }

        bsc_hv_t input[HV_SEGMENTS];
        for (int s = 0; s < HV_SEGMENTS; s++) {
            bsc_bindN(input[s], query_vectors[s]);
        }

        //tracer::write_input(query_inds, input);

        size_t pred_inds[RN_FEATURES];
        size_t iter = 0;
        rn(
            pred_inds,
            iter,
            input,
            initial_estimates,
            cbs,
            max_iter,
            convergence_threshold,
            topaPT_threshold
        );

        // Check HLS RN output
        uint64_t correct = 0;
        for (auto f = 0; f < query_inds.size(); f++) {
            if (query_inds[f] == pred_inds[f]) {
                correct++;
            }
        }
        correct_factors += correct;
        correct_frames += correct == RN_FEATURES ? 1 : 0;

        // Register number of iterations for convergence
        niters.emplace_back(static_cast<float>(iter));
    }

    float acc_frames = static_cast<float>(correct_frames) / static_cast<float>(queries.size());
    float acc_factors = static_cast<float>(correct_factors) / static_cast<float>((RN_FEATURES*queries.size()));
    std::cout << "acc_frame: " << acc_frames << ", acc_factor: " << acc_factors << std::endl;

    float niter_avg = mean(niters);
    float niter_min = *std::min_element(niters.begin(), niters.end());
    float niter_max = *std::max_element(niters.begin(), niters.end());
    float niter_std = stdev(niters);
    std::cout << "niter_avg: " << niter_avg << " niter_std: " << niter_std << " niter_min: " << niter_min << " niter_max: " << niter_max << " max_iter: " << max_iter << std::endl;

    return 0;
}
