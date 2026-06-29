#include "bind.hpp"

void hello(int &out, const int &a, const int &b) {
    out = a ^ b;
}

void my_bind(bsc_hv_t &out, bsc_hv_t &a, bsc_hv_t &b) {
    bsc_bind(out, a, b);
}

