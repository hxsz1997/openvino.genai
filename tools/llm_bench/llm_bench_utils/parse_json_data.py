# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


def parse_text_json_data(json_data_list):
    text_param_list = []
    for json_data in json_data_list:
        if 'prompt' in json_data:
            if json_data['prompt'] != '':
                text_param_list.append(json_data['prompt'])
            else:
                raise RuntimeError('== prompt should not be empty string ==')
        else:
            raise RuntimeError('== key word "prompt" does not exist ==')
    return text_param_list


def parse_image_json_data(json_data_list):
    image_param_list = []
    for data in json_data_list:
        image_param = {}
        if 'prompt' in data:
            if data['prompt'] != '':
                image_param['prompt'] = data['prompt']
            else:
                raise RuntimeError('== prompt should not be empty string ==')
        else:
            raise RuntimeError('== key word "prompt" does not exist in prompt file ==')
        if 'width' in data:
            image_param['width'] = int(data['width'])
        if 'height' in data:
            image_param['height'] = int(data['height'])
        if 'steps' in data:
            image_param['steps'] = int(data['steps'])
        if 'guidance_scale' in data:
            image_param['guidance_scale'] = float(data['guidance_scale'])
        image_param_list.append(image_param)
    return image_param_list


def parse_speech_json_data(json_data_list):
    speech_param_list = []
    for json_data in json_data_list:
        speech_param = {}
        if 'media' in json_data:
            if json_data['media'] != '':
                speech_param['media'] = json_data['media']
            else:
                raise RuntimeError('== media path should not be empty string ==')
        else:
            raise RuntimeError('== key word "media" does not exist ==')
        if 'language' in json_data:
            speech_param['language'] = json_data['language']
        if 'timestamp' in json_data:
            speech_param['timestamp'] = json_data['timestamp']
        speech_param_list.append(speech_param)
    return speech_param_list


def parse_multimodal_json_data(json_data_list):
    multimodal_param_list = []
    for json_data in json_data_list:
        if 'image' in json_data:
            if json_data['image'] == '':
                raise RuntimeError('== image should not be empty string ==')
        else:
            raise RuntimeError(f'== key word "image" does not exist ==')
        if 'text' in json_data:
            if json_data['text'] == '':
                raise RuntimeError('== text should not be empty string ==')
        else:
            raise RuntimeError(f'== key word "text" does not exist ==')
        multimodal_param_list.append(json_data)
    return multimodal_param_list