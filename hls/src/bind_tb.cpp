#include <iostream>

#include "bind.hpp"
#include "hyle/bsc.hpp"

int main(int argc, char *argv[]) {
    std::cout << "Hello World" << std::endl;
    bsc_hv_t a =        {1, 0, 1, 0, 1, 1, 1, 1, 1, 1};
    bsc_hv_t b =        {0, 1, 0, 1, 1, 0, 0, 1, 1, 0};
    bsc_hv_t out_gold = {1, 1, 1, 1, 0, 1, 1, 0, 0, 1};
    bsc_hv_t out;
    my_bind(out, a, b);
    std::cout << out << std::endl;
}
