#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import numpy as np
import openvino_genai
from PIL import Image
from openvino import Tensor
from pathlib import Path


def streamer(subword: str) -> bool:
    '''

    Args:
        subword: sub-word of the generated text.

    Returns: Return flag corresponds whether generation should be stopped.

    '''
    print(subword, end='', flush=True)

    # No value is returned as in this example we don't want to stop the generation in this method.
    # "return None" will be treated the same as "return False".


def read_image(path: str) -> Tensor:
    '''

    Args:
        path: The path to the image.

    Returns: the ov.Tensor containing the image.

    '''
    pic = Image.open(path).convert("RGB")
    image_data = np.array(pic.getdata()).reshape(1, 3, pic.size[1], pic.size[0]).astype(np.byte)
    return Tensor(image_data)


def read_images(path: str) -> list[Tensor]:
    entry = Path(path)
    if entry.is_dir():
        return [read_image(str(file)) for file in sorted(entry.iterdir())]
    return [read_image(path)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('image_dir', help="Image file or dir with images")
    args = parser.parse_args()

    rgbs = read_images(args.image_dir)

    device = 'CPU'  # GPU can be used as well
    enable_compile_cache = dict()
    if "GPU" == device:
        # Cache compiled models on disk for GPU to save time on the
        # next run. It's not beneficial for CPU.
        enable_compile_cache["CACHE_DIR"] = "vlm_cache"
    pipe = openvino_genai.VLMPipeline(args.model_dir, device, **enable_compile_cache)

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 100

    pipe.start_chat()
    prompt = 'What is in the picture?'
    generation_result = pipe.generate(prompt, images=rgbs, generation_config=config, streamer=streamer)
    print("generation_result: ", generation_result.texts)
    perf_metrics = generation_result.perf_metrics
    raw_metrics = perf_metrics.raw_metrics
    print("detokenization_durations:", raw_metrics.detokenization_durations)
    print("generate_durations:", raw_metrics.generate_durations)
    print("m_batch_sizes:", raw_metrics.m_batch_sizes)
    print("m_durations:", raw_metrics.m_durations)
    print("m_times_to_first_token:", raw_metrics.m_times_to_first_token)
    print("tokenization_durations:", raw_metrics.tokenization_durations)
    print(f"Load time: {perf_metrics.get_load_time():.2f} ms")
    print(f"Generate time: {perf_metrics.get_generate_duration().mean:.2f} ± {perf_metrics.get_generate_duration().std:.2f} ms")
    print(f"Tokenization time: {perf_metrics.get_tokenization_duration().mean:.2f} ± {perf_metrics.get_tokenization_duration().std:.2f} ms")
    print(f"Detokenization time: {perf_metrics.get_detokenization_duration().mean:.2f} ± {perf_metrics.get_detokenization_duration().std:.2f} ms")
    # while True:
    #     try:
    #         prompt = input("\n----------\n"
    #             "question:\n")
    #     except EOFError:
    #         break
    #     pipe.generate(prompt, generation_config=config, streamer=streamer)
    pipe.finish_chat()


if '__main__' == __name__:
    main()
