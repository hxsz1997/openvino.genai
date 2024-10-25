# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import time
import logging as log
import llm_bench_utils.ov_utils
import llm_bench_utils.pt_utils
import llm_bench_utils.model_utils as model_utils
import numpy as np
import hashlib
import llm_bench_utils.metrics_print as metrics_print
import llm_bench_utils.output_csv
from transformers import set_seed
import llm_bench_utils.output_json
import llm_bench_utils.output_file
import llm_bench_utils.gen_output_data as gen_output_data
from PIL import Image
from openvino import Tensor
from pathlib import Path

FW_UTILS = {'pt': llm_bench_utils.pt_utils, 'ov': llm_bench_utils.ov_utils}

DEFAULT_OUTPUT_TOKEN_SIZE = 512

def run_minicpmv2(input_text, num, model, tokenizer, args, iter_data_list, md5_list, 
                  prompt_index, bench_hook, model_precision, proc_id, mem_consumption):
    print("======run_minicpmv2======")
    set_seed(args['seed'])
    input_text_list = [input_text] * args['batch_size']
    if args["output_dir"] is not None and num == 0:
        for bs_index, in_text in enumerate(input_text_list):
            llm_bench_utils.output_file.output_input_text(in_text, args, model_precision, prompt_index, bs_index, proc_id)
    image = input_text['image']
    image = Image.open(image)
    question = input_text['text']
    msgs = [{"role": "user", "content": question}]
    max_rss_mem_consumption = ''
    max_uss_mem_consumption = ''
    max_shared_mem_consumption = ''
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.start_collect_memory_consumption()
    max_gen_tokens = DEFAULT_OUTPUT_TOKEN_SIZE if args['infer_count'] is None else args['infer_count']
    if args['infer_count'] is not None and args['end_token_stopping'] is False:
        # model.generation_config.eos_token_id = None
        model.config.eos_token_id = None
        res, tok_encode_time, input_token_size, tok_decode_time, generation_time, generated_token_size = model.chat(image=image, msgs=msgs, context=None, tokenizer=tokenizer, sampling=False, stream=False, max_new_tokens=int(max_gen_tokens), eos_token_id=None)
    else:
        res, tok_encode_time, input_token_size, tok_decode_time, generation_time, generated_token_size = model.chat(image=image, msgs=msgs, context=None, tokenizer=tokenizer, sampling=False, stream=False, max_new_tokens=int(max_gen_tokens))

    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.end_collect_momory_consumption()
        max_rss_mem_consumption, max_shared_mem_consumption, max_uss_mem_consumption = mem_consumption.get_max_memory_consumption()
        mem_consumption.clear_max_memory_consumption()

    result_md5_list = []
    for bs_idx in range(args['batch_size']):    
        if args["output_dir"] is not None:
            llm_bench_utils.output_file.output_gen_text(res, args, model_precision, prompt_index, num, bs_idx, proc_id)
        result_md5_list.append(hashlib.new("md5", res.encode(), usedforsecurity=False).hexdigest())
    if len(md5_list[num]) == 0:
        md5_list[num] = {prompt_index : result_md5_list}
    else:
        md5_list[num][prompt_index] = result_md5_list
    per_token_time = ""
    if generated_token_size > 0:
        per_token_time = generation_time / generated_token_size
    else:
        log.warning("No generated tokens")    
    
    
    print("=========tok_encode_time=========", tok_encode_time)
    print("=========input_token_size=========", input_token_size)
    print("=========tok_decode_time=========", tok_decode_time)
    print("=========generation_time=========", generation_time)
    print("=========generated_token_size=========", generated_token_size)
    print("=========res=========", res)
    
    tm_list = []
    tm_infer_list = []
    vision_list = []
    sampler_list = []
    tm_infer_list.extend(model.get_llm_times()[0])
    tm_list.extend(model.get_llm_times()[1])
    vision_list.extend(model.get_llm_times()[2])
    sampler_list.extend(model.get_llm_times()[3])

    print("==========len(tm_list)=========", len(tm_list))
    
    iter_data = gen_output_data.gen_iterate_data(
        iter_idx=num,
        loop_idx='',
        in_size=input_token_size * args['batch_size'],
        infer_count=len(tm_infer_list),
        out_size=generated_token_size,
        gen_time=generation_time /1000,
        latency=per_token_time,
        res_md5=result_md5_list,
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        max_uss_mem=max_uss_mem_consumption,
        prompt_idx=prompt_index,
        tokenization_time=(tok_encode_time, tok_decode_time),
        vision_latency=vision_list[0]*1000,
        sampler_latency=sampler_list[0]*1000
    )
    iter_data_list.append(iter_data)
    metrics_print.print_metrics(
        num,
        iter_data,
        tm_list,
        tm_infer_list,
        warm_up=(num == 0),
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        max_uss_mem=max_uss_mem_consumption,
        tokenization_time=(tok_encode_time, tok_decode_time),
        batch_size=args['batch_size'],
        prompt_idx=prompt_index
    )
    if num > 0:
        prev_md5 = md5_list[num - 1][prompt_index]
        if result_md5_list != prev_md5:
            log.warning(f"[{num}] Prompt[{prompt_index}]'s md5 {result_md5_list} "
                        f"is different from md5 of the {num - 1} iteration {prev_md5}")
            metrics_print.print_generated(num, warm_up=(num == 0), generated=res, prompt_idx=prompt_index)
            if not args.get("use_cb", False):
                if num == 1:
                    # if the device is CPU, throw exception
                    if args['devices'].lower().startswith('cpu') is True:
                        assert (result_md5_list == prev_md5)
                else:
                    # throw exception
                    assert (result_md5_list == prev_md5)
    else:
        metrics_print.print_generated(num, warm_up=(num == 0), generated=res, prompt_idx=prompt_index)

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

def run_minicpmv2_genai(input_text, num, model, tokenizer, args, iter_data_list, md5_list, prompt_index,
                        model_precision, proc_id, mem_consumption, config):
    set_seed(args['seed'])
    input_text_list = [input_text] * args['batch_size']
    if args["output_dir"] is not None and num == 0:
        for bs_index, in_text in enumerate(input_text_list):
            llm_bench_utils.output_file.output_input_text(in_text, args, model_precision, prompt_index, bs_index, proc_id)
    image = input_text['image']
    rgbs = read_images(image)
    prompt = input_text['text']
    # generation_result = model.generate(prompt, images=rgbs, generation_config=config)
    # print("generation_result: ", generation_result.texts)
    # pt_inputs = tokenizer(input_text_list, return_tensors="pt")
    # input_token_size = pt_inputs.input_ids.shape[1]
    # if args['batch_size'] > 1:
    #     out_str = '[warm-up]' if num == 0 else '[{}]'.format(num)
    #     out_str += " Batch_size={}, ".format(args['batch_size'])
    #     out_str += 'all input token size after padding: {} * {}, '.format(input_token_size, args['batch_size'])
    #     if args['infer_count'] is not None:
    #         out_str += 'all max_output_token_size: {} * {}'.format(args['infer_count'], args['batch_size'])
    #     log.info(out_str)

    max_rss_mem_consumption = ''
    max_uss_mem_consumption = ''
    max_shared_mem_consumption = ''
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.start_collect_memory_consumption()
    config.max_new_tokens = DEFAULT_OUTPUT_TOKEN_SIZE if args['infer_count'] is None else args['infer_count']
    start = time.perf_counter()
    generation_result = model.generate(prompt, images=rgbs, generation_config=config)
    end = time.perf_counter()
    generated_text = generation_result.texts
    perf_metrics = generation_result.perf_metrics

    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.end_collect_momory_consumption()
        max_rss_mem_consumption, max_shared_mem_consumption, max_uss_mem_consumption = mem_consumption.get_max_memory_consumption()
        mem_consumption.clear_max_memory_consumption()

    generation_time = end - start
    generated_tokens = [tokenizer(text).input_ids for text in generated_text]
    num_tokens = 0
    result_md5_list = []
    for bs_idx in range(args['batch_size']):
        generated_text_len = len(generated_tokens[bs_idx])
        num_tokens += generated_text_len
        if generated_text_len > config.max_new_tokens:
            log.error('Output token size is over max output token size!')
        result_text = generated_text[bs_idx]
        if args["output_dir"] is not None:
            llm_bench_utils.output_file.output_gen_text(result_text, args, model_precision, prompt_index, num, bs_idx, proc_id)
        result_md5_list.append(hashlib.new("md5", result_text.encode(), usedforsecurity=False).hexdigest())
    if len(md5_list[num]) == 0:
        md5_list[num] = {prompt_index : result_md5_list}
    else:
        md5_list[num][prompt_index] = result_md5_list
    per_token_time = ""
    if num_tokens > 0:
        per_token_time = generation_time * 1000 / (num_tokens / args['batch_size'])
    else:
        log.warning("No generated tokens")
    print("num_tokens: ", num_tokens)
    print("generated_text: ", generated_text)
    tm_list = np.array(perf_metrics.raw_metrics.m_durations) / 1000 / 1000
    log.debug('latency of all tokens:')
    [log.debug('[{}]{:.4f}'.format(idx, tm)) for idx, tm in enumerate(tm_list)]
    tokenization_time = (
        np.mean(perf_metrics.raw_metrics.tokenization_durations) / 1000,
        np.mean(perf_metrics.raw_metrics.detokenization_durations) / 1000
    )
    iter_data = gen_output_data.gen_iterate_data(
        iter_idx=num,
        loop_idx='',
        in_size=0,
        infer_count=len(tm_list),
        out_size=num_tokens,
        gen_time=generation_time,
        latency=per_token_time,
        res_md5=result_md5_list,
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        max_uss_mem=max_uss_mem_consumption,
        prompt_idx=prompt_index,
        tokenization_time=tokenization_time
    )
    iter_data_list.append(iter_data)
    metrics_print.print_metrics(
        num,
        iter_data,
        tm_list.tolist(),
        [],
        warm_up=(num == 0),
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        max_uss_mem=max_uss_mem_consumption,
        tokenization_time=tokenization_time,
        batch_size=args['batch_size'],
        prompt_idx=prompt_index
    )
    if num > 0:
        prev_md5 = md5_list[num - 1][prompt_index]
        if result_md5_list != prev_md5:
            log.warning(f"[{num}] Prompt[{prompt_index}]'s md5 {result_md5_list} "
                        f"is different from md5 of the {num - 1} iteration {prev_md5}")
            metrics_print.print_generated(num, warm_up=(num == 0), generated=generated_text[0], prompt_idx=prompt_index)
            if not args.get("use_cb", False):
                if num == 1:
                    # if the device is CPU, throw exception
                    if args['devices'].lower().startswith('cpu') is True:
                        assert (result_md5_list == prev_md5)
                else:
                    # throw exception
                    assert (result_md5_list == prev_md5)
    else:
        metrics_print.print_generated(num, warm_up=(num == 0), generated=generated_text[0], prompt_idx=prompt_index)


def run_minicpmv2_benchmark(model_path, framework, device, args, num_iters, mem_consumption):
    if framework =='pt':
        raise RuntimeError('== Minicpmv2 is not support pt framework ==')
    if args['genai']:
        model, pretrain_time, config, tokenizer = FW_UTILS[framework].create_genai_minicpmv2_model(model_path, device, **args)
    else:
        model, tokenizer, pretrain_time = FW_UTILS[framework].create_minicpmv2_model(model_path, device, **args)
    print("============ from_pretrained_time ==========", pretrain_time)
    model_precision = llm_bench_utils.model_utils.get_model_precision(model_path.parts)
    iter_data_list = []
    md5_list = {num : {} for num in range(num_iters + 1)}
    input_text_list = llm_bench_utils.model_utils.get_multimodal_param_from_prompt_file(args)
    if args['prompt_index'] is None:
        prompt_idx_list = [prompt_idx for prompt_idx, input_text in enumerate(input_text_list)]
        text_list = input_text_list
    else:
        prompt_idx_list = []
        text_list = []
        for i in args['prompt_index']:
            if 0 <= i < len(input_text_list):
                text_list.append(input_text_list[i])
                prompt_idx_list.append(i)
    if len(input_text_list) == 0:
        raise RuntimeError('==Failure prompts is empty ==')
    log.info(f'Benchmarking iter nums(exclude warm-up): {num_iters}, prompt nums: {len(text_list)}, '
             f"prompt idx: {prompt_idx_list}, num_beams: {args['num_beams']}")

    proc_id = os.getpid()
    bench_hook = None
    if args['subsequent'] is False:
        for num in range(num_iters + 1):
            for idx, input_text in enumerate(text_list):
                if num == 0:
                    log.info(f'[warm-up] Input text: {input_text}')
                if args['genai']:    
                    run_minicpmv2_genai(input_text, num, model, tokenizer, args, iter_data_list, md5_list, prompt_idx_list[idx], model_precision, proc_id, mem_consumption, config)
                else:
                    run_minicpmv2(input_text, num, model, tokenizer, args, iter_data_list, md5_list, prompt_idx_list[idx], bench_hook, model_precision, proc_id, mem_consumption)
    else:
        for idx, input_text in enumerate(text_list):
            for num in range(num_iters + 1):
                if num == 0:
                    log.info(f'[warm-up] Input text: {input_text}')
                if args['genai']:    
                    run_minicpmv2_genai(input_text, num, model, tokenizer, args, iter_data_list, md5_list, prompt_idx_list[idx], model_precision, proc_id, mem_consumption, config)
                else:
                    run_minicpmv2(input_text, num, model, tokenizer, args, iter_data_list, md5_list, prompt_idx_list[idx], bench_hook, model_precision, proc_id, mem_consumption)

    llm_bench_utils.metrics_print.print_average(iter_data_list, prompt_idx_list, args['batch_size'], True)
    return iter_data_list, pretrain_time
