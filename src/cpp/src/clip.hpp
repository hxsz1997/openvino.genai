// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <numeric>

//#define CLIP_DEBUG_FUNCTIONS
enum projector_type {
    PROJECTOR_TYPE_RESAMPLER,
    PROJECTOR_TYPE_UNKNOWN,
};

struct clip_ctx {
    bool has_text_encoder = false;
    bool has_vision_encoder = false;
    bool has_minicpmv_projector = false;

    float image_mean[3];
    float image_std[3];
    int32_t ftype = 1;

    std::vector<uint8_t> buf_compute_meta;

    projector_type proj_type = PROJECTOR_TYPE_RESAMPLER;
    size_t patch_size = 0;
    size_t image_size = 0;
};

// RGB uint8 image
struct clip_image_u8 {
    int nx;
    int ny;

    std::vector<uint8_t> buf;
};

// RGB float32 image (NHWC)
// Memory layout: RGBRGBRGB...
struct clip_image_f32 {
    int nx;
    int ny;

    std::vector<float> buf;
};

/** interpret bytes as an image file with length bytes_length, and use the result to populate img */
bool clip_image_load_from_bytes(const unsigned char * bytes, size_t bytes_length, struct clip_image_u8 * img);

bool bicubic_resize(const clip_image_u8& img, clip_image_u8& dst, int target_width, int target_height);

/** preprocess img and store the result in res_imgs, pad_to_square may be overriden to false depending on model configuration */
clip_image_f32 clip_image_preprocess(struct clip_ctx& ctx, const clip_image_u8& img);
