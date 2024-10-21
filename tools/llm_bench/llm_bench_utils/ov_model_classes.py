# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# flake8: noqa
import time
import inspect
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple, Callable, Iterable, Any
from tempfile import TemporaryDirectory
import PIL
import numpy as np
import torch
from diffusers.schedulers import LMSDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import PIL_INTERPOLATION
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from optimum.intel.openvino import OVModelForCausalLM
from optimum.intel.openvino.utils import ONNX_WEIGHTS_NAME, OV_XML_FILE_NAME
from openvino.runtime import Model, Core, Tensor, Type
from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
from transformers import GenerationConfig, StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList, LogitsProcessor
from transformers.generation.utils import GenerateOutput
from threading import Thread
from copy import deepcopy
import json
from PIL import Image
from transformers import AutoProcessor, TextIteratorStreamer
from transformers.generation import GenerationMixin
from transformers import AutoConfig, GenerationConfig
import openvino as ov
from openvino.runtime.passes import Manager, MatcherPass, WrapType, Matcher
from openvino.runtime import opset13



class OVMPTModel(OVModelForCausalLM):
    def _reshape(
        self,
        model: Model,
        *args,
        **kwargs,
    ):
        shapes = {}
        for inputs in model.inputs:
            shapes[inputs] = inputs.get_partial_shape()
            shapes[inputs][0] = -1
            if shapes[inputs].rank.get_length() in [2, 3]:
                shapes[inputs][1] = -1
            else:
                if '.key' in inputs.get_any_name():
                    shapes[inputs][3] = -1
                elif inputs.get_any_name() != "beam_idx":
                    shapes[inputs][2] = -1
        model.reshape(shapes)
        return model


    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        self.compile()

        if self.use_cache and past_key_values is not None:
            input_ids = input_ids[:, -1:]

        batch_size = input_ids.shape[0]

        inputs = {}
        past_len = 0
        if not self.stateful:
            if past_key_values is not None:
                past_len = past_key_values[0][1].shape[-2]
                if self._pkv_precision == Type.bf16:
                    # numpy does not support bf16, pretending f16, should change to bf16
                    past_key_values = tuple(
                        Tensor(past_key_value, past_key_value.shape, Type.bf16)
                        for pkv_per_layer in past_key_values
                        for past_key_value in pkv_per_layer
                    )
                else:
                    # Flatten the past_key_values
                    past_key_values = tuple(
                        past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer
                    )
                

                # Add the past_key_values to the decoder inputs
                inputs = dict(zip(self.key_value_input_names, past_key_values))

            # Create empty past_key_values for decoder_with_past first generation step
            elif self.use_cache:
                for input_name in self.key_value_input_names:
                    model_inputs = self.model.input(input_name)
                    shape = model_inputs.get_partial_shape()
                    if self.config.model_type == 'chatglm':
                        shape[0] = 0
                        shape[1] = batch_size
                    else:
                        shape[0] = batch_size
                        if shape[2].is_dynamic:
                            shape[2] = 0
                        elif shape.rank.get_length() == 4 and shape[3].is_dynamic:
                            shape[3] = 0
                        else:
                            shape[1] = 0
                    inputs[input_name] = Tensor(model_inputs.get_element_type(), shape.get_shape())
        else:
            # past_key_values are not used explicitly, instead they are handled inside the model
            if past_key_values is None:
                # Need a marker to differentiate the first generate iteration from the others in
                # the first condition at the function beginning above.
                # It should be something that is not None and it should be True when converted to Boolean.
                past_key_values = ((),)
                # This is the first iteration in a sequence, reset all states
                for state in self.request.query_state():
                    state.reset()
                # Set initial value for the next beam_idx input that will be used at the current iteration
                # and will be optionally updated by _reorder_cache at the next iterations if beam_search is used
                self.next_beam_idx = np.array(range(batch_size), dtype=int)

        inputs["input_ids"] = np.array(input_ids)
        # Add the attention_mask inputs when needed
        if "attention_mask" in self.input_names or "position_ids" in self.input_names:
            if attention_mask is not None:
                attention_mask = np.array(attention_mask)
            else:
                attention_mask = np.ones(
                    (input_ids.shape[0], input_ids.shape[1] + past_len), dtype=inputs["input_ids"].dtype
                )

        if "attention_mask" in self.input_names:
            inputs["attention_mask"] = attention_mask

        if "position_ids" in self.input_names:
            if position_ids is not None:
                position_ids = np.array(position_ids)
            else:
                position_ids = np.cumsum(attention_mask, axis=1) - 1
                position_ids[attention_mask == 0] = 1
                if past_key_values:
                    position_ids = np.expand_dims(position_ids[:, -1], axis=-1)

            inputs["position_ids"] = position_ids

        if hasattr(self, 'next_beam_idx') and "beam_idx" in self.input_names:
            inputs['beam_idx'] = self.next_beam_idx

        # Run inference
        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()
        logits = torch.from_numpy(self.request.get_tensor("logits").data).to(self.device)

        if not self.stateful:
            if self.use_cache:
                # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the self-attention layer)
                past_key_values = tuple(self.request.get_tensor(key).data for key in self.key_value_output_names)
                # Tuple of tuple of length `n_layers`, with each tuple of length equal to 2 (k/v of self-attention)
                past_key_values = tuple(
                    past_key_values[i : i + self.num_pkv] for i in range(0, len(past_key_values), self.num_pkv)
                )
            else:
                past_key_values = None

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)


class OVLDMSuperResolutionPipeline(DiffusionPipeline):
    def __init__(self, model_path: Path, core: Core, device: str):
        super().__init__()
        self.vqvae = core.compile_model(model_path / 'vqvae.xml', device)
        self.unet = core.compile_model(model_path / 'unet.xml', device)
        self.scheduler = LMSDiscreteScheduler.from_config(model_path / 'scheduler_config.json')
        self._unet_output = self.unet.output(0)
        self._vqvae_output = self.vqvae.output(0)

    @torch.no_grad()
    def __call__(
        self,
        image: Union[torch.Tensor, PIL.Image.Image] = None,
        batch_size: Optional[int] = 1,
        num_inference_steps: Optional[int] = 100,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = 'pil',
        return_dict: bool = True,
        tm_list: Optional[List] = None,
        **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:
        r'''
        Args:
            image (`torch.Tensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            batch_size (`int`, *optional*, defaults to 1):
                Number of images to generate.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            output_type (`str`, *optional*, defaults to `'pil'`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        '''
        image = image

        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        else:
            raise ValueError(f'`image` has to be of type `PIL.Image.Image` or `torch.Tensor` but is {type(image)}')

        if isinstance(image, PIL.Image.Image):
            image = self.preprocess(image)

        height, width = image.shape[-2:]

        # in_channels should be 6: 3 for latents, 3 for low resolution image
        latents_shape = (batch_size, 3, height, width)
        latents = randn_tensor(latents_shape, generator=generator)
        # set timesteps and move to the correct device
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps_tensor = self.scheduler.timesteps
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        latents = latents.numpy()
        extra_kwargs = {}
        if 'eta' in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_kwargs['eta'] = eta

        for t in timesteps_tensor:
            # concat latents and low resolution image in the channel dimension.
            latents_input = np.concatenate([latents, image], axis=1)
            latents_input = self.scheduler.scale_model_input(latents_input, t)
            # predict the noise residual
            tic = time.perf_counter()
            noise_pred = self.unet([latents_input, t])[self._unet_output]
            tm_list.append(time.perf_counter() - tic)
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(torch.from_numpy(noise_pred), t, torch.from_numpy(latents))['prev_sample'].numpy()

        # decode the image latents with the VQVAE
        tic = time.perf_counter()
        image = self.vqvae(latents)[self._vqvae_output]
        tm_list.append(time.perf_counter() - tic)
        image = image / 2 + 0.5
        image = image.transpose(0, 2, 3, 1)

        if output_type == 'pil':
            image = self.numpy_to_pil(image)
        return image

    @staticmethod
    def preprocess(image):
        w, h = image.size
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        image = image.resize((w, h), resample=PIL_INTERPOLATION['lanczos'])
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2.0 * image - 1.0


class OVChatGLMModel(OVModelForCausalLM):
    position_encoding_2d = True
    num_layers = 28
    max_sequence_length = 128
    bos_token_id = 130004
    eos_token_id = 130005
    mask_token_id = 130000
    gmask_token_id = 130001

    def __init__(
        self,
        model: Model,
        config: PretrainedConfig = None,
        device: str = 'CPU',
        dynamic_shapes: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        super().__init__(model, config, device, dynamic_shapes, ov_config, model_save_dir, **kwargs)
        self.is_v1 = False
        if not self.stateful and not self.key_value_input_names:
            self.is_v1 = True
            self.key_value_input_names = ['past_key_values']
            self.key_value_output_names = [o.any_name for o in self.model.outputs[1:]]

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        if not self.is_v1:
            return super().prepare_inputs_for_generation(
                input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask,
                position_ids=position_ids,
                past=past,
                **kwargs
            )
        batch_size, seq_length = input_ids.shape
        mask = self.mask_token_id
        g_mask = self.gmask_token_id
        seqs = input_ids.tolist()
        mask_positions, use_gmasks = [], []
        for seq in seqs:
            tmp_mask_token = g_mask if g_mask in seq else mask
            use_gmask = tmp_mask_token == g_mask
            mask_positions.append(seq.index(tmp_mask_token))
            use_gmasks.append(use_gmask)

        # only last token for input_ids if past is not None
        if past is not None or past_key_values is not None:
            # Next Step Inference
            last_token = input_ids[:, -1].unsqueeze(-1)
            # if attention_mask is not None:
            if attention_mask is not None and attention_mask.dtype == torch.bool:
                attention_mask = attention_mask[:, :, -1:]
            else:
                attention_mask = None
            if position_ids is not None:
                position_ids = position_ids[..., -1:]
            else:
                context_lengths = [seq.index(self.bos_token_id) for seq in seqs]
                if self.position_encoding_2d:  # position_encoding_2d = True
                    position_ids = torch.tensor(
                        [[mask_position, seq_length - context_length] for mask_position, context_length in zip(mask_positions, context_lengths)],
                        dtype=torch.long,
                        device=input_ids.device,
                    ).unsqueeze(-1)
                else:
                    position_ids = torch.tensor([mask_position for mask_position in mask_positions], dtype=torch.long, device=input_ids.device).unsqueeze(-1)

            if past is None:
                past = self.get_past_key_values(past_key_values)
            return {
                'input_ids': last_token,
                'past_key_values': past,
                'position_ids': position_ids,
                'attention_mask': attention_mask,
                'use_cache': self.use_cache,
                'token_type_ids': None,
            }
        else:
            # First Step Inference
            if attention_mask is not None and attention_mask.dtype != torch.bool:
                attention_mask = None
            if attention_mask is None:
                attention_mask = self.get_masks(
                    input_ids,
                    device=input_ids.device,
                )
            if position_ids is None:
                position_ids = self.get_position_ids(
                    input_ids,
                    device=input_ids.device,
                    mask_positions=mask_positions,
                    use_gmasks=use_gmasks,
                )
            past_key_values = None
            if self.use_cache:
                past_key_values = np.zeros((self.num_layers, 2, 0, 1, 32, 128))
                # numpy does not support bf16, pretending f16, should change to bf16
                if self._pkv_precision == Type.bf16:
                    past_key_values = Tensor(past_key_values, past_key_values.shape, Type.bf16)
            return {
                'input_ids': input_ids,
                'position_ids': position_ids,
                'attention_mask': attention_mask,
                'past_key_values': past_key_values,
                'use_cache': self.use_cache,
                'token_type_ids': None,
            }

    def get_masks(self, input_ids, device):
        batch_size, seq_length = input_ids.shape
        context_lengths = [seq.tolist().index(self.bos_token_id) for seq in input_ids]
        attention_mask = torch.ones((batch_size, seq_length, seq_length), device=device)
        attention_mask.tril_()
        for i, context_length in enumerate(context_lengths):
            attention_mask[i, :, :context_length] = 1
        attention_mask.unsqueeze_(1)
        attention_mask = (attention_mask < 0.5).bool()

        return attention_mask

    def get_position_ids(self, input_ids, mask_positions, device, use_gmasks=None):
        batch_size, seq_length = input_ids.shape
        if use_gmasks is None:
            use_gmasks = [False] * batch_size
        context_lengths = [seq.tolist().index(self.bos_token_id) for seq in input_ids]
        if self.position_encoding_2d:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
            for i, context_length in enumerate(context_lengths):
                position_ids[i, context_length:] = mask_positions[i]
            block_position_ids = [
                torch.cat(
                    (
                        torch.zeros(context_length, dtype=torch.long, device=device),
                        torch.arange(seq_length - context_length, dtype=torch.long, device=device) + 1,
                    )
                )
                for context_length in context_lengths
            ]
            block_position_ids = torch.stack(block_position_ids, dim=0)
            position_ids = torch.stack((position_ids, block_position_ids), dim=1)
        else:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
            for i, context_length in enumerate(context_lengths):
                if not use_gmasks[i]:
                    position_ids[context_length:] = mask_positions[i]

        return position_ids

    @staticmethod
    def get_past_key_values(pkv):
        pkv_combined = []
        for i in range(0, len(pkv)):
            past_key_values_pair = np.stack(pkv[i], axis=0)
            pkv_combined.append(past_key_values_pair)
        pkv_combined = np.array(pkv_combined)
        return pkv_combined

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        
        if not self.is_v1:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values, **kwargs)
        self.compile()

        inputs = {}
        if past_key_values is not None:
            inputs['past_key_values'] = past_key_values
        inputs['input_ids'] = np.array(input_ids)

        # Add the attention_mask inputs when needed
        if 'attention_mask' in self.input_names and attention_mask is not None:
            inputs['attention_mask'] = np.array(attention_mask)

        if 'position_ids' in kwargs and kwargs['position_ids'] is not None:
            inputs['position_ids'] = np.array(kwargs['position_ids'])

        # Run inference
        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()

        logits = torch.from_numpy(self.request.get_tensor('logits').data).to(self.device)

        if self.use_cache:
            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the self-attention layer)
            past_key_values = tuple(self.request.get_tensor(key).data for key in self.key_value_output_names)
            # Tuple of tuple of length `n_layers`, with each tuple of length equal to 2 (k/v of self-attention)
            past_key_values = tuple(past_key_values[i : i + self.num_pkv] for i in range(0, len(past_key_values), self.num_pkv))
        else:
            past_key_values = None

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    def _reshape(
        self,
        model: Model,
        batch_size: int,
        sequence_length: int,
        height: int = None,
        width: int = None,
    ):
        return model


class InsertSlice(MatcherPass):
    def __init__(self):
        MatcherPass.__init__(self)
        self.model_changed = False

        param = WrapType("opset10.Result")

        def callback(matcher: Matcher) -> bool:
            root = matcher.get_match_root()
            if root is None:
                return False
            if len(root.get_output_partial_shape(0)) == 3:
                parent = root.input_value(0).get_node()
                grand_parent = parent.input_value(0).get_node()

                grand_parent_output = parent.input(0).get_source_output()
                consumers = grand_parent_output.get_target_inputs()
                start = np.array([0, -1, 0], dtype=np.int32)
                stop = np.array([1, -2, grand_parent_output.get_partial_shape()[-1].get_length()], dtype=np.int32)
                step = np.array([1, -1, 1], dtype=np.int32)
                axes = np.array([0, 1, 2], dtype=np.int32)
                slice = opset13.slice(grand_parent, start, stop, step, axes, name="inserted_slice")
                for consumer in consumers:
                    consumer.replace_source_output(slice.output(0))
                self.model_changed = True
                # Use new operation for additional matching
                self.register_new_node(slice)
                print("applied slice for lm head")

                return True

        self.register_matcher(Matcher(param, "InsertSlice"), callback)


def get_2d_sincos_pos_embed(embed_dim, image_size):
    """
    image_size: image_size or (image_height, image_width)
    return:
    pos_embed: [image_height, image_width, embed_dim]
    """
    if isinstance(image_size, int):
        grid_h_size, grid_w_size = image_size, image_size
    else:
        grid_h_size, grid_w_size = image_size[0], image_size[1]

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[0])  # (H, W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[1])  # (H, W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=-1)  # (H, W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_new(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (H, W)
    out: (H, W, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    out = np.einsum("hw,d->hwd", pos, omega)  # (H, W, D/2), outer product

    emb_sin = np.sin(out)  # (H, W, D/2)
    emb_cos = np.cos(out)  # (H, W, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (H, W, D)
    return emb


def prepare_vis_position_ids(pixel_values, patch_attention_mask, tgt_sizes, patch_size, num_patches_per_side):
    batch_size = pixel_values.size(0)
    max_im_h, max_im_w = pixel_values.size(2), pixel_values.size(3)
    max_nb_patches_h, max_nb_patches_w = max_im_h // patch_size, max_im_w // patch_size
    boundaries = torch.arange(1 / num_patches_per_side, 1.0, 1 / num_patches_per_side)
    position_ids = torch.full(size=(batch_size, max_nb_patches_h * max_nb_patches_w), fill_value=0)

    for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
        if tgt_sizes is not None:
            nb_patches_h = tgt_sizes[batch_idx][0]
            nb_patches_w = tgt_sizes[batch_idx][1]
        else:
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0].sum()

        fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
        fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

        bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
        bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

        pos_ids = (bucket_coords_h[:, None] * num_patches_per_side + bucket_coords_w).flatten()
        position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

    return position_ids

class OvModelForCausalLMWithEmb(GenerationMixin):
    def __init__(self, core, model_dir, device="CPU", ov_config=None, compile=True, slice_lm_head=True) -> None:
        self.core = core
        self._supports_cache_class = False
        self.config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        self.config.is_decoder = True
        self.config.is_encoder_decoder = False
        self.generation_config = GenerationConfig.from_model_config(self.config)
        model_dir = Path(model_dir)
        self.model = core.read_model(model_dir / "language_model.xml")
        self.token_emb = core.read_model(model_dir / "embed_tokens.xml")
        if slice_lm_head:
            self.slice_lm_head()
        self.request = None
        self.token_emb_request = None
        self._device = device.upper()
        self.device = torch.device("cpu")
        self.ov_config = ov_config
        self.next_beam_idx = None
        self._past_length = None
        self.input_names = [input_t.get_any_name() for input_t in self.model.inputs]
        self.main_input_name = "input_ids"
        self.llm_times = []
        self.tm_list = []
        if compile:
            self.compile()

    def slice_lm_head(self):
        manager = Manager()
        manager.register_pass(InsertSlice())
        manager.run_passes(self.model)
        self.model.validate_nodes_and_infer_types()

    def compile(self):
        if self.request is None:
            self.request = self.core.compile_model(self.model, self._device, self.ov_config).create_infer_request()
        self._compile_token_emb()

    def _compile_token_emb(self):
        if self.token_emb_request is None:
            self.token_emb_request = self.core.compile_model(self.token_emb, self._device, self.ov_config)

    def to(self, device: str):
        if isinstance(device, str):
            self._device = device.upper()
            self.clear_requests()

        return self

    def clear_requests(self):
        del self.request
        del self.token_emb_request
        self.request = None
        self.token_emb_request = None

    def embed_tokens(self, input_ids: torch.LongTensor):
        self._compile_token_emb()
        res = self.token_emb_request(input_ids, share_inputs=True)
        return res[0]

    def prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]

        inputs = {}
        # past_key_values are not used explicitly, instead they are handled inside the model
        if past_key_values is None:
            self.llm_times = []
            self.tm_list = []
            # This is the first iteration in a sequence, reset all states
            if self.request is not None:
                self.request.reset_state()
                # Set initial value for the next beam_idx input that will be used at the current iteration
                # and will be optionally updated by _reorder_cache at the next iterations if beam_search is used
                self.next_beam_idx = np.arange(batch_size, dtype=int)
                self._past_length = 0
        past_len = self._get_past_length(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids if past_key_values is None else input_ids[:, -1:])

            if hasattr(self.config, "scale_emb"):
                inputs_embeds = inputs_embeds * self.config.scale_emb
        inputs["inputs_embeds"] = inputs_embeds

        # Add the attention_mask inputs when needed
        if "attention_mask" in self.input_names or "position_ids" in self.input_names:
            if attention_mask is not None:
                attention_mask = np.array(attention_mask)
            else:
                attention_mask = np.ones((inputs_embeds.shape[0], inputs_embeds.shape[1] + past_len), dtype=int)

        if "attention_mask" in self.input_names:
            inputs["attention_mask"] = attention_mask

        if "position_ids" in self.input_names:
            if position_ids is not None:
                position_ids = np.array(position_ids)
            else:
                position_ids = np.cumsum(attention_mask, axis=1) - 1
                position_ids[attention_mask == 0] = 1
                if past_key_values:
                    position_ids = position_ids[:, -input_ids.shape[1] :]

            inputs["position_ids"] = position_ids

        if "beam_idx" in self.input_names:
            inputs["beam_idx"] = self.next_beam_idx if self.next_beam_idx is not None else np.arange(batch_size, dtype=int)

        return inputs

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        self.compile()

        tic = time.perf_counter()
        inputs = self.prepare_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        start = time.perf_counter()
        # Run inference
        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()
        self.llm_times.append(time.perf_counter() - start)
        logits = self.request.get_tensor("logits").data
        logits = torch.from_numpy(logits).to(self.device)
        past_key_values = ((),)
        self._past_length += inputs["inputs_embeds"].shape[1]

        self.tm_list.append(time.perf_counter() - tic)

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    # Adapted from transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        attention_mask = kwargs.get("attention_mask", None)
        use_cache = kwargs.get("use_cache", None)

        if past_key_values is not None:
            past_len = self._get_past_length(past_key_values)
            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and input_ids is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_len) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif input_ids is not None and past_len < input_ids.shape[1]:
                input_ids = input_ids[:, past_len:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None and "position_ids" in self.input_names:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values and input_ids is not None:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds if past_key_values is None else None,
        }

        return model_inputs

    def _get_past_length(self, past_key_values=None):
        if past_key_values is None:
            return 0
        return self._past_length

    # Adapted from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel._reorder_cache
    def _reorder_cache(self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called.
        This is required to match `past_key_values` with the correct beam_idx at every generation step.
        """
        self.next_beam_idx = np.array(beam_idx)  # save beam_idx to be used as an input in the next iteration
        return past_key_values

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""

        return True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_llm_times(self):
        return self.llm_times, self.tm_list


class OvMiniCPMV:
    def __init__(self, config, vpm, resampler, llm, processor):
        self.config = config
        self.llm = llm
        self.vpm = vpm
        self.embed_dim = self.llm.config.hidden_size
        self._resampler = resampler
        self.processor = processor
        self._pos_embeds = torch.from_numpy(get_2d_sincos_pos_embed(self.embed_dim, 70)).float()
        self.max_size = (70, 70)
        self.vpm_times = []
        self.resampler_times = []

        self.terminators = ["<|im_end|>", "<|endoftext|>"]

    def set_decoder(self, decoder):
        self.llm = decoder

    def get_decoder(self):
        return self.llm

    def resampler(self, x, tgt_sizes):
        bs = x.shape[0]

        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]

        self._adjust_pos_cache(tgt_sizes)

        max_patch_len = torch.max(patch_len)
        key_padding_mask = torch.zeros((bs, max_patch_len), dtype=torch.bool)

        pos_embed = []
        for i in range(bs):
            tgt_h, tgt_w = tgt_sizes[i]
            pos_embed.append(self._pos_embeds[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)))  # patches * D
            key_padding_mask[i, patch_len[i] :] = True

        pos_embed = torch.nn.utils.rnn.pad_sequence(pos_embed, batch_first=True, padding_value=0.0).permute(1, 0, 2)  # BLD => L * B * D

        start = time.perf_counter()
        res = torch.from_numpy(self._resampler([x, pos_embed, key_padding_mask])[0])
        self.resampler_times.append(time.perf_counter() - start)
        return res

    def _set_2d_pos_cache(self, max_size):
        pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(self.embed_dim, max_size)).float()
        self._pos_embed = pos_embed

    def _adjust_pos_cache(self, tgt_sizes):
        max_h = torch.max(tgt_sizes[:, 0])
        max_w = torch.max(tgt_sizes[:, 1])
        if max_h > self.max_size[0] or max_w > self.max_size[1]:
            self.max_size = [max(max_h, self.max_size[0]), max(max_w, self.max_size[1])]
            self._set_2d_pos_cache(self.max_size)

    def get_vllm_embedding(self, data):
        if "vision_hidden_states" not in data:
            tgt_sizes = data["tgt_sizes"]
            pixel_values_list = data["pixel_values"]
            vision_hidden_states = []
            all_pixel_values = []
            img_cnt = []
            for pixel_values in pixel_values_list:
                img_cnt.append(len(pixel_values))
                all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

            # exist image
            if all_pixel_values:
                tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]
                tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

                max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

                all_pixel_values = torch.nn.utils.rnn.pad_sequence(all_pixel_values, batch_first=True, padding_value=0.0)
                B, L, _ = all_pixel_values.shape
                all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

                patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool)
                for i in range(B):
                    patch_attn_mask[i, 0, : tgt_sizes[i][0] * tgt_sizes[i][1]] = True

                vision_batch_size = 32
                all_pixel_values = all_pixel_values
                if B > vision_batch_size:
                    hs = []
                    for i in range(0, B, vision_batch_size):
                        start_idx = i
                        end_idx = i + vision_batch_size
                        block_pxl_values = all_pixel_values[start_idx:end_idx]
                        block_patch_attn_mask = patch_attn_mask[start_idx:end_idx]
                        block_tgt_sizes = tgt_sizes[start_idx:end_idx]
                        block_position_ids = prepare_vis_position_ids(
                            block_pxl_values,
                            block_patch_attn_mask,
                            block_tgt_sizes,
                            self.config.vision_config.patch_size,
                            self.config.vision_config.image_size // self.config.patch_size,
                        )
                        start = time.perf_counter()
                        tmp_hs = torch.from_numpy(self.vpm([block_pxl_values, block_patch_attn_mask, block_position_ids])[0])
                        self.vpm_times.append(time.perf_counter() - start)
                        hs.append(tmp_hs)
                    vision_embedding = torch.cat(hs, dim=0)
                else:
                    position_ids = prepare_vis_position_ids(
                        all_pixel_values,
                        patch_attn_mask,
                        tgt_sizes,
                        self.config.vision_config.patch_size,
                        self.config.vision_config.image_size // self.config.patch_size,
                    )
                    start = time.perf_counter()
                    vision_embedding = torch.from_numpy(self.vpm([all_pixel_values, patch_attn_mask, position_ids])[0])
                    self.vpm_times.append(time.perf_counter() - start)
                vision_embedding = self.resampler(vision_embedding, tgt_sizes)

                start = 0
                for pixel_values in pixel_values_list:
                    img_cnt = len(pixel_values)
                    if img_cnt > 0:
                        vision_hidden_states.append(vision_embedding[start : start + img_cnt])
                        start += img_cnt
                    else:
                        vision_hidden_states.append([])
            else:  # no image
                dummy_feature = []
                for _ in range(len(pixel_values_list)):
                    vision_hidden_states.append(dummy_feature)

        else:
            vision_hidden_states = data["vision_hidden_states"]

        if hasattr(self.llm.config, "scale_emb"):
            vllm_embedding = self.llm.embed_tokens(data["input_ids"]) * self.llm.config.scale_emb
        else:
            vllm_embedding = self.llm.embed_tokens(data["input_ids"])

        bs = len(data["input_ids"])
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            if len(cur_vs_hs) > 0:
                cur_vllm_emb = torch.from_numpy(vllm_embedding[i])
                cur_image_bound = data["image_bound"][i]
                if len(cur_image_bound) > 0:
                    image_indices = torch.stack([torch.arange(r[0], r[1], dtype=torch.long) for r in cur_image_bound])

                    cur_vllm_emb.scatter_(0, image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]), cur_vs_hs.view(-1, cur_vs_hs.shape[-1]))
        return vllm_embedding

    def forward(self, data, **kwargs):
        vllm_embedding = self.get_vllm_embedding(data)
        position_ids = data["position_ids"]
        if position_ids.dtype != torch.int64:
            position_ids = position_ids.long()

        return self.llm(input_ids=None, position_ids=position_ids, inputs_embeds=vllm_embedding, **kwargs)

    def _decode(self, inputs_embeds, tokenizer, attention_mask, decode_text=False, **kwargs):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        output = self.llm.generate(
            inputs_embeds=torch.from_numpy(inputs_embeds), pad_token_id=0, eos_token_id=terminators, attention_mask=attention_mask, **kwargs
        )
        print("=======output=====", output.shape)
        if decode_text:
            tok_decode_start = time.perf_counter()
            result_text = self._decode_text(output, tokenizer)
            tok_decode_end = time.perf_counter()
            tok_decode_time = (tok_decode_end - tok_decode_start) * 1000
            return result_text, tok_decode_time
        return output

    def _decode_stream(self, inputs_embeds, tokenizer, **kwargs):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        streamer = TextIteratorStreamer(tokenizer=tokenizer)
        generation_kwargs = {"inputs_embeds": torch.from_numpy(inputs_embeds), "pad_token_id": 0, "eos_token_id": terminators, "streamer": streamer}
        generation_kwargs.update(kwargs)

        thread = Thread(target=self.llm.generate, kwargs=generation_kwargs)
        thread.start()

        return streamer

    def _decode_text(self, result_ids, tokenizer):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        result_text = []
        for result in result_ids:
            result = result[result != 0]
            if result[0] == tokenizer.bos_id:
                result = result[1:]
            if result[-1] in terminators:
                result = result[:-1]
            result_text.append(tokenizer.decode(result).strip())
        return result_text

    def generate(
        self,
        input_ids=None,
        pixel_values=None,
        tgt_sizes=None,
        image_bound=None,
        attention_mask=None,
        tokenizer=None,
        vision_hidden_states=None,
        return_vision_hidden_states=False,
        stream=False,
        decode_text=False,
        **kwargs,
    ):
        assert input_ids is not None
        assert len(input_ids) == len(pixel_values)

        model_inputs = {
            "input_ids": input_ids,
            "image_bound": image_bound,
        }

        if vision_hidden_states is None:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["tgt_sizes"] = tgt_sizes
        else:
            model_inputs["vision_hidden_states"] = vision_hidden_states

        with torch.inference_mode():
            model_inputs["inputs_embeds"] = self.get_vllm_embedding(model_inputs)

            if stream:
                result = self._decode_stream(model_inputs["inputs_embeds"], tokenizer, **kwargs)
            else:
                result, tok_decode_time = self._decode(model_inputs["inputs_embeds"], tokenizer, attention_mask, decode_text=decode_text, **kwargs)

        return result, tok_decode_time

    def chat(
        self,
        image,
        msgs,
        tokenizer,
        processor=None,
        vision_hidden_states=None,
        max_new_tokens=2048,
        min_new_tokens=0,
        sampling=True,
        max_inp_length=8192,
        system_prompt="",
        stream=False,
        max_slice_nums=None,
        use_image_id=None,
        **kwargs,
    ):
        self.vpm_times = []
        self.resampler_times = []
        if isinstance(msgs[0], list):
            batched = True
        else:
            batched = False
        msgs_list = msgs
        images_list = image

        if batched is False:
            images_list, msgs_list = [images_list], [msgs_list]
        else:
            assert images_list is None, "Please integrate image to msgs when using batch inference."
            images_list = [None] * len(msgs_list)
        assert len(images_list) == len(msgs_list), "The batch dim of images_list and msgs_list should be the same."

        if processor is None:
            if self.processor is None:
                self.processor = AutoProcessor.from_pretrained(self.config._name_or_path, trust_remote_code=True)
            processor = self.processor

        assert (
            self.config.query_num == processor.image_processor.image_feature_size
        ), "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert (
            self.config.patch_size == processor.image_processor.patch_size
        ), "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert (
            self.config.use_image_id == processor.image_processor.use_image_id
        ), "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert (
            self.config.slice_config.max_slice_nums == processor.image_processor.max_slice_nums
        ), "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert (
            self.config.slice_mode == processor.image_processor.slice_mode
        ), "These two values should be the same. Check `config.json` and `preprocessor_config.json`."

        prompts_lists = []
        input_images_lists = []
        for image, msgs in zip(images_list, msgs_list):
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            copy_msgs = deepcopy(msgs)

            assert len(msgs) > 0, "msgs is empty"

            if image is not None and isinstance(copy_msgs[0]["content"], str):
                copy_msgs[0]["content"] = [image, copy_msgs[0]["content"]]

            images = []
            for i, msg in enumerate(copy_msgs):
                role = msg["role"]
                content = msg["content"]
                assert role in ["user", "assistant"]
                if i == 0:
                    assert role == "user", "The role of first msg should be user"
                if isinstance(content, str):
                    content = [content]
                cur_msgs = []
                for c in content:
                    if isinstance(c, Image.Image):
                        images.append(c)
                        cur_msgs.append("(<image>./</image>)")
                    elif isinstance(c, str):
                        cur_msgs.append(c)
                msg["content"] = "\n".join(cur_msgs)

            if system_prompt:
                sys_msg = {"role": "system", "content": system_prompt}
                copy_msgs = [sys_msg] + copy_msgs

            prompts_lists.append(processor.tokenizer.apply_chat_template(copy_msgs, tokenize=False, add_generation_prompt=True))
            input_images_lists.append(images)

        tok_encode_start = time.perf_counter()
        inputs = processor(
            prompts_lists, input_images_lists, max_slice_nums=max_slice_nums, use_image_id=use_image_id, return_tensors="pt", max_length=max_inp_length
        )
        tok_encode_end = time.perf_counter()
        tok_encode_time = (tok_encode_end - tok_encode_start) * 1000
        input_token_size = inputs['input_ids'][0].numel()
        # print("=========input_token_size=========", inputs['input_ids'][0].numel())

        if sampling:
            generation_config = {"top_p": 0.8, "top_k": 100, "temperature": 0.7, "do_sample": True, "repetition_penalty": 1.05}
        else:
            generation_config = {
                "repetition_penalty": 1.2,
            }

        if min_new_tokens > 0:
            generation_config["min_new_tokens"] = min_new_tokens

        generation_config.update((k, kwargs[k]) for k in generation_config.keys() & kwargs.keys())

        inputs.pop("image_sizes")
        with torch.inference_mode():
            res, tok_decode_time = self.generate(
                **inputs,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                vision_hidden_states=vision_hidden_states,
                stream=stream,
                decode_text=True,
                **generation_config,
            )

        if stream:

            def stream_gen():
                for text in res:
                    for term in self.terminators:
                        text = text.replace(term, "")
                    yield text

            return stream_gen(), tok_encode_time, input_token_size

        else:
            if batched:
                answer = res
            else:
                answer = res[0]
            return answer, tok_encode_time, input_token_size, tok_decode_time
    
    def get_llm_times(self):
        tm_infer_list, tm_list = self.llm.get_llm_times()
        return tm_infer_list, tm_list, self.vpm_times, self.resampler_times


def init_model(core, model_dir, llm_model_dir, image_emb_path, resampler_path, device):
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    llm = OvModelForCausalLMWithEmb(core, model_dir / llm_model_dir, device)
    img_emb = core.compile_model(model_dir / image_emb_path, device)
    resampler = core.compile_model(model_dir / resampler_path, device)
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

    ov_model = OvMiniCPMV(config, img_emb, resampler, llm, processor)
    tokenizer = ov_model.processor.tokenizer
    return ov_model, tokenizer

