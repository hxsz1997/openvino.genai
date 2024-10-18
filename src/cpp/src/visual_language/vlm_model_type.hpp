// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>

#include "openvino/genai/visibility.hpp"
#include <openvino/core/except.hpp>

namespace ov::genai {

enum class OPENVINO_GENAI_EXPORTS VLMModelType {
    MINICPM,
    LLAVA,
    LLAVA_NEXT,
};

inline VLMModelType to_vlm_model_type(const std::string& value) {
    static const std::unordered_map<std::string, VLMModelType> model_types_map = {
        {"minicpmv", VLMModelType::MINICPM},
        {"llava", VLMModelType::LLAVA},
        {"llava_next", VLMModelType::LLAVA_NEXT}
    };

    auto it = model_types_map.find(value);
    if (it != model_types_map.end()) {
        return it->second;
    }
    OPENVINO_THROW("Unsupported '", value, "' VLM model type");
}
}