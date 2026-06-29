#pragma once

#include "rn.hpp"
#include "hyle/common.hpp"
#include "hyle/vsa.hpp"
#include "hyle/dataset.hpp"

namespace tracer {
    void init(const char *path);
    void set_query(int query);
    void set_iter(int iter);
    void set_feature(int feature);
    void write_input(const data_t &inds, const hv_t &input);
    void write_unbind(const hv_t &output, const hv_t &input, const bsc_hv_t (&features)[RN_FEATURES-1]);
    void write_similarity(const hls::vector<int, RN_CODEBOOK_SIZE> &sim);
}
