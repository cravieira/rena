#include "prng.hpp"

#include <cstdint>

uint32_t xs32(uint32_t state) {
    uint32_t x = state;

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;

    return x;
}

uint32_t _state = 12031;

