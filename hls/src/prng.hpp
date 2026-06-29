#pragma once

#include <hls_vector.h>

#include <cstdint>
#include <cstddef>


uint32_t xs32(uint32_t state);

template<int N>
void rand_noise(hls::vector<int, N> (&out), uint32_t &final_state, uint32_t seed) {
    uint32_t state = seed;
    int noise_max=32;
    for (int i = 0; i < N; i++) {
        uint32_t new_seed = xs32(state);
        uint32_t x = new_seed % (noise_max*2); // TODO: This should be improved for flexible max noise values and reduce hardware costs
        x -= noise_max;
        out[i] = static_cast<int>(x);
        state = new_seed;
    }
    final_state = state;
}
