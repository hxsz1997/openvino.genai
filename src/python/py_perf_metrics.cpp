// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>

#include "openvino/genai/perf_metrics.hpp"

namespace py = pybind11;

using ov::genai::MeanStdPair;
using ov::genai::PerfMetrics;
using ov::genai::RawPerfMetrics;

namespace {

auto raw_perf_metrics_docstring = R"(
    Structure with raw performance metrics for each generation before any statistics are calculated.

    :param generate_durations: Durations for each generate call in microseconds.
    :type generate_durations: List[MicroSeconds]

    :param tokenization_durations: Durations for the tokenization process in microseconds.
    :type tokenization_durations: List[MicroSeconds]

    :param detokenization_durations: Durations for the detokenization process in microseconds.
    :type detokenization_durations: List[MicroSeconds]

    :param m_times_to_first_token: Times to the first token for each call in microseconds.
    :type m_times_to_first_token: List[MicroSeconds]

    :param m_new_token_times: Time points for each new token generated.
    :type m_new_token_times: List[TimePoint]

    :param m_batch_sizes: Batch sizes for each generate call.
    :type m_batch_sizes: List[int]

    :param m_durations: Total durations for each generate call in microseconds.
    :type m_durations: List[MicroSeconds]

    :param num_generated_tokens: Total number of tokens generated.
    :type num_generated_tokens: int

    :param num_input_tokens: Total number of tokens in the input prompt.
    :type num_input_tokens: int
)";

auto perf_metrics_docstring = R"(
    Holds performance metrics for each generate call.

    PerfMetrics holds fields with mean and standard deviations for the following metrics:
    - Time To the First Token (TTFT), ms
    - Time per Output Token (TPOT), ms/token
    - Generate total duration, ms
    - Tokenization duration, ms
    - Detokenization duration, ms
    - Throughput, tokens/s

    Additional fields include:
    - Load time, ms
    - Number of generated tokens
    - Number of tokens in the input prompt

    Preferable way to access values is via get functions. Getters calculate mean and std values from raw_metrics and return pairs.
    If mean and std were already calculated, getters return cached values.

    :param get_load_time: Returns the load time in milliseconds.
    :type get_load_time: float

    :param get_num_generated_tokens: Returns the number of generated tokens.
    :type get_num_generated_tokens: int

    :param get_num_input_tokens: Returns the number of tokens in the input prompt.
    :type get_num_input_tokens: int

    :param get_ttft: Returns the mean and standard deviation of TTFT in milliseconds.
    :type get_ttft: MeanStdPair

    :param get_tpot: Returns the mean and standard deviation of TPOT in milliseconds.
    :type get_tpot: MeanStdPair

    :param get_throughput: Returns the mean and standard deviation of throughput in tokens per second.
    :type get_throughput: MeanStdPair

    :param get_generate_duration: Returns the mean and standard deviation of generate durations in milliseconds.
    :type get_generate_duration: MeanStdPair

    :param get_tokenization_duration: Returns the mean and standard deviation of tokenization durations in milliseconds.
    :type get_tokenization_duration: MeanStdPair

    :param get_detokenization_duration: Returns the mean and standard deviation of detokenization durations in milliseconds.
    :type get_detokenization_duration: MeanStdPair

    :param raw_metrics: A structure of RawPerfMetrics type that holds raw metrics.
    :type raw_metrics: RawPerfMetrics
)";

template <typename T, typename U>
std::vector<float> get_ms(const T& instance, U T::*member) {
    // Converts c++ duration to float so that it can be used in Python.
    std::vector<float> res;
    const auto& durations = instance.*member;
    res.reserve(durations.size());
    std::transform(durations.begin(), durations.end(), std::back_inserter(res),
                   [](const auto& duration) { return duration.count(); });
    return res;
}

} // namespace

void init_perf_metrics(py::module_& m) {
    py::class_<RawPerfMetrics>(m, "RawPerfMetrics", raw_perf_metrics_docstring)
        .def(py::init<>())
        .def_property_readonly("generate_durations", [](const RawPerfMetrics &rw) {
            return get_ms(rw, &RawPerfMetrics::generate_durations);
        })
        .def_property_readonly("tokenization_durations", [](const RawPerfMetrics &rw) { 
            return get_ms(rw, &RawPerfMetrics::tokenization_durations);
        })
        .def_property_readonly("detokenization_durations", [](const RawPerfMetrics &rw) { 
            return get_ms(rw, &RawPerfMetrics::detokenization_durations); 
        })
        .def_property_readonly("m_times_to_first_token", [](const RawPerfMetrics &rw) {
            return get_ms(rw, &RawPerfMetrics::m_times_to_first_token);
        })
        .def_property_readonly("m_durations", [](const RawPerfMetrics &rw) {
            return get_ms(rw, &RawPerfMetrics::m_durations);
        })
        .def_readonly("m_batch_sizes", &RawPerfMetrics::m_batch_sizes);

    py::class_<MeanStdPair>(m, "MeanStdPair")
        .def(py::init<>())
        .def_readonly("mean", &MeanStdPair::mean)
        .def_readonly("std", &MeanStdPair::std)
        .def("__iter__", [](const MeanStdPair &self) {
            return py::make_iterator(&self.mean, &self.std + 1);
        }, py::keep_alive<0, 1>());  // Keep object alive while the iterator is used;

    py::class_<PerfMetrics>(m, "PerfMetrics", perf_metrics_docstring)
        .def(py::init<>())
        .def("get_load_time", &PerfMetrics::get_load_time)
        .def("get_num_generated_tokens", &PerfMetrics::get_num_generated_tokens)
        .def("get_num_input_tokens", &PerfMetrics::get_num_input_tokens)
        .def("get_ttft", &PerfMetrics::get_ttft)
        .def("get_tpot", &PerfMetrics::get_tpot)
        .def("get_ipot", &PerfMetrics::get_ipot)
        .def("get_throughput", &PerfMetrics::get_throughput)
        .def("get_generate_duration", &PerfMetrics::get_generate_duration)
        .def("get_inference_duration", &PerfMetrics::get_inference_duration)
        .def("get_tokenization_duration", &PerfMetrics::get_tokenization_duration)
        .def("get_detokenization_duration", &PerfMetrics::get_detokenization_duration)
        .def("__add__", &PerfMetrics::operator+)
        .def("__iadd__", &PerfMetrics::operator+=)
        .def_readonly("raw_metrics", &PerfMetrics::raw_metrics);
}
