#include "tracer.hpp"

// Tracing is available only for testbenching
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <hls_vector.h>

#include <filesystem>

// Control variables
bool _tracing = false;
std::string _path;
int _query = 0;
int _iter = 0;
int _feature = 0;

// Auxiliary functions
// Pretty printer for hls::vector of similarities
template<size_t N>
std::ostream& operator<<(std::ostream& os, const hls::vector<int, N> c) {
    for (auto it = c.cbegin(); it != c.cend(); it++) {
        //os << std::hex << std::setw(1) << std::setfill('0') << *it;
        os << *it << "";
    }

    return os;
}

template<typename T>
int _write_vector(const std::string &path, const std::vector<T> &v) {
    std::ofstream f(path);

    // Check if the file was opened successfully
    if (!f.is_open()) {
        std::cerr << "Error opening file "+path+"!" << std::endl;
        return 1; // Indicate an error
    }

    std::ostream_iterator<T> output_iterator(f, "");
    std::copy(std::begin(v), std::end(v), output_iterator);
    f << std::endl;
    f.close();
    return 0;
}

template<typename T, size_t N>
int _write_hls_vector(const std::string &path, const hls::vector<T, N> v) {
    std::ofstream f(path);

    // Check if the file was opened successfully
    if (!f.is_open()) {
        std::cerr << "Error opening file "+path+"!" << std::endl;
        return 1; // Indicate an error
    }

    f << v << std::endl;
    f.close();
    return 0;
}

int _write_hv(const std::string &path, const hv_t &hv) {
    std::ofstream f(path);

    // Check if the file was opened successfully
    if (!f.is_open()) {
        std::cerr << "Error opening file "+path+"!" << std::endl;
        return 1; // Indicate an error
    }

    f << hv << std::endl;
    f.close();
    return 0;
}

void tracer::init(const char *path) {
    _path = std::string(path);
    std::filesystem::create_directories(_path);
    _tracing = true;
}

void tracer::set_query(int query) {
    _query = query;
}

void tracer::set_iter(int iter) {
    _iter = iter;
}

void tracer::set_feature(int feature) {
    _feature = feature;
}

void tracer::write_input(const data_t &inds, const hv_t &input) {
    if (!_tracing) {
        return;
    }

    std::string query_p = std::to_string(_query);
    std::string iter_p = std::to_string(_iter);
    std::string feature_p = std::to_string(_feature);
    std::string p = _path+"/q_"+query_p+"/input";
    std::filesystem::create_directories(p);

    _write_hv(p+"/vector.txt", input);
    _write_vector(p+"/inds.txt", inds);
}

void tracer::write_unbind(const hv_t &output, const hv_t &input, const bsc_hv_t (&features)[RN_FEATURES-1]) {
    if (!_tracing) {
        return;
    }

    std::string query_p = std::to_string(_query);
    std::string iter_p = std::to_string(_iter);
    std::string feature_p = std::to_string(_feature);
    std::string p = _path+"/q_"+query_p+"/i_"+iter_p+"/f_"+feature_p+"/unbind";
    std::filesystem::create_directories(p);

    _write_hv(p+"/output.txt", output);
    _write_hv(p+"/input.txt", input);
    for (int i = 0; i < RN_FEATURES-1; i++) {
        _write_hv(p+"/f_"+std::to_string(i)+".txt", features[i]);
    }
}

void tracer::write_similarity(const hls::vector<int, RN_CODEBOOK_SIZE> &sim) {
    if (!_tracing) {
        return;
    }

    std::string query_p = std::to_string(_query);
    std::string iter_p = std::to_string(_iter);
    std::string feature_p = std::to_string(_feature);
    std::string p = _path+"/q_"+query_p+"/i_"+iter_p+"/f_"+feature_p+"/similarity";
    std::filesystem::create_directories(p);

    _write_hls_vector<int, RN_CODEBOOK_SIZE>(p+"/sim.txt", sim);
}
