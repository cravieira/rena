#include <cstddef>
#include <iostream>

#include "rn.hpp"
#include "hyle/bsc.hpp"

#include <hls_vector.h>

// Pretty printer for hls::vector
template<typename T, size_t N>
std::ostream& operator<<(std::ostream& os, const hls::vector<T, N> c) {
    for (auto it = c.cbegin(); it != c.cend(); it++) {
        //os << std::hex << std::setw(1) << std::setfill('0') << *it;
        os << *it << " ";
    }

    return os;
}

int test_unbind() {
    bsc_hv_t s = {1, 0, 1, 0, 1, 1, 1, 1, 1, 1};
    bsc_hv_t features[RN_FEATURES-1] = {
        {0, 1, 0, 1, 1, 0, 0, 1, 0, 1},
        {1, 1, 1, 1, 1, 1, 1, 1, 0, 1}
    };
    const bsc_hv_t *features_ptr_arr[RN_FEATURES-1] = { &features[0], &features[1] };
    bsc_hv_t out_gold = {0, 0, 0, 0, 1, 0, 0, 1, 1, 1};
    bsc_hv_t out;
    unbind_module(out, s, features_ptr_arr);
    if (out != out_gold) {
        std::cout << s << std::endl;
        std::cout << features[0] << std::endl;
        std::cout << features[1] << std::endl;
        std::cout << out << std::endl;
        return 1;
    };

    return 0;
}

int test_hammdist_to_dotsim() {
    hls::vector<int, 10> out;
    bsc_dist_t hamm[10] =           { 0, 1, 2, 3, 4, 5, 6, 7, 8, 10};
    hls::vector<int, 10> out_gold = { 5, 4, 3, 2, 1, 0,-1,-2,-3, -5};
    hammdist_to_dotsim<10>(out, hamm);

    if (out != out_gold) {
        std::cout << out << std::endl;
        std::cout << out_gold << std::endl;
        return 1;
    };
    return 0;
}

int test_hv_sign() {
    bsc_hv_t out;
    hls::vector<int, 10> acc = {-5, -4, 3, -2, 1, 0, -1, 2, -3, 5};
    bsc_hv_t out_gold        = { 0,  0, 1,  0, 1, 0,  0, 1,  0, 1};
    hv_sign(out, acc);

    if (out != out_gold) {
        std::cout << out << std::endl;
        std::cout << out_gold << std::endl;
        return 1;
    };
    return 0;
}

int test_vector_projection() {
    bsc_hv_t codebook[RN_CODEBOOK_SIZE] = {
        {0, 0, 1, 0, 0, 0, 1, 1, 0, 1},
        {1, 1, 1, 0, 1, 1, 1, 1, 0, 0},
        {1, 0, 1, 0, 1, 0, 0, 0, 0, 1},
        {0, 0, 1, 1, 1, 0, 0, 1, 0, 1}
    };
    hls::vector<int, RN_CODEBOOK_SIZE> dot_dists = {3, -2, 1, 0};
    hls::vector<int, HV_SEGMENT_SIZE> out_gold = {-4, -6, 2, -2, -4, -6, 0, 0, -2, 6};
    hls::vector<int, HV_SEGMENT_SIZE> out;

    vector_projection(out, dot_dists, codebook);

    if (out != out_gold) {
        std::cout << out << std::endl;
        std::cout << out_gold << std::endl;
        return 1;
    };
    return 0;
}

int test_rn_dp() {
    bsc_hv_t codebook[RN_CODEBOOK_SIZE] = {
        {0, 0, 1, 0, 0, 0, 1, 1, 0, 1},
        {1, 1, 1, 0, 1, 1, 1, 1, 0, 0},
        {1, 0, 1, 0, 1, 0, 0, 0, 0, 1},
        {0, 0, 1, 1, 1, 0, 0, 1, 0, 1}
    };
    bsc_hv_t s = {0, 1, 1, 0, 1, 1, 0, 1, 0, 1};
    bsc_hv_t y = {0, 1, 1, 0, 0, 1, 1, 0, 1, 0};
    bsc_hv_t z = {1, 0, 0, 1, 1, 0, 1, 1, 0, 1};
    const bsc_hv_t *features[RN_FEATURES-1] = { &y, &z };
    bsc_hv_t out_gold = {0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    bsc_hv_t out;

    rn_dp(out, s, features, codebook);

    if (out != out_gold) {
        std::cout << out << std::endl;
        std::cout << out_gold << std::endl;
        return 1;
    };
    return 0;
}

int test_rn() {
    bsc_hv_t s = {0, 1, 1, 0, 1, 1, 0, 1, 0, 1};
    bsc_hv_t estimates[RN_FEATURES] = {
        {1, 0, 1, 0, 1, 0, 1, 1, 0, 1},
        {0, 1, 1, 0, 0, 1, 1, 0, 1, 0},
        {1, 0, 0, 1, 1, 0, 1, 1, 0, 1}
    };
    bsc_hv_t codebooks[RN_FEATURES][RN_CODEBOOK_SIZE] = {
        {
        {0, 0, 1, 0, 0, 0, 1, 1, 0, 1},
        {1, 1, 1, 0, 1, 1, 1, 1, 0, 0},
        {1, 0, 1, 0, 1, 0, 0, 0, 0, 1},
        {0, 0, 1, 1, 1, 0, 0, 1, 0, 1}
        },

        {
        {1, 1, 0, 1, 0, 1, 0, 1, 0, 0},
        {0, 1, 1, 0, 0, 1, 0, 0, 1, 0},
        {0, 1, 1, 0, 0, 1, 1, 0, 1, 0},
        {0, 0, 1, 0, 1, 1, 1, 1, 1, 1}
        },

        {
        {1, 1, 1, 1, 1, 0, 0, 1, 0, 1},
        {1, 0, 0, 1, 0, 1, 1, 1, 0, 0},
        {1, 0, 0, 0, 1, 0, 0, 0, 0, 1},
        {0, 1, 0, 1, 0, 1, 1, 1, 0, 1}
        }
    };
    size_t iter = 0;
    size_t max_iter = 3;
    hls::vector<size_t, RN_FEATURES> out_gold = {0, 2, 3};
    size_t out[RN_FEATURES];

    rn(out, iter, s, estimates, codebooks, max_iter);

    hls::vector<size_t, RN_FEATURES> out_vec;
    for (size_t f = 0; f < RN_FEATURES; f++) { out_vec[f] = out[f]; }
    if (out_vec != out_gold) {
        std::cout << out_vec << std::endl;
        std::cout << out_gold << std::endl;
        return 1;
    };
    return 0;
}

int main(int argc, char *argv[]) {
    test_unbind();
    test_hammdist_to_dotsim();
    test_hv_sign();
    test_vector_projection();
    test_rn_dp();
    test_rn();
}
