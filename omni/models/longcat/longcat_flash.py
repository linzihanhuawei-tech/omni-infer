# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Longcat-Flash model."""
from typing import Iterable, List, Optional, Set, Tuple, Union
import torch
import torch.distributed as dist
from torch import nn
from transformers import PretrainedConfig
torch._logging.set_logs(recompiles=True)
# vllm adaptor
from vllm.config import CacheConfig, QuantizationConfig, VllmConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.attention import AttentionMetadata
from vllm.sequence import IntermediateTensors
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    tensor_model_parallel_all_gather
)
from omni.adaptors.vllm.distributed.parallel_state import (
    get_mlp_tp_group,
    get_o_proj_tp_group,
    GroupCoordinator
)

from vllm.distributed.parallel_state import (
    get_tp_group
)

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.utils import (
    PPMissingLayer, 
    is_pp_missing_parameter, 
    make_layers, 
    make_empty_intermediate_tensors_factory,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from omni.models.common.layers.attention.deepseek_mla import DeepseekMLA
from omni.models.common.layers.rotary_embedding import RotaryEmbedding2

from omni.models.common.layers.vocab_parallel_embedding import (
    ParallelLMHead, 
    VocabParallelEmbedding
)
from omni.models.common.layers.layernorm import RMSNorm

from omni.models.common.layers.moe.fused_moe.layer import FusedMoE
from omni.models.common.config.model_config import model_extra_config
from omni.models.longcat.longcat_moe import LongcatFlashMoE
from omni.models.longcat.longcat_mla import LongcatFlashMLA
from omni.models.common.layers.fused_mlp.layer import FusedMLP
from omni.models.deepseek.deepseek_v3 import ParallelDeepseekMLP
if model_extra_config.operator_opt_config.unquant_bmm_nz:
    # if use weight nz, this config must be True
    torch.npu.config.allow_internal_format = True

"""MLP module activation split length, split by 64G VRAM, need to confirm the optimal split length based on sequence length and performance"""
SEQ_SPLIT_LENGTH_BEFORE_ALL_GATHER = 64


def rank0_log(msg):
    if get_tensor_model_parallel_rank() == 0:
        print(msg)
import inspect
def check_stack_for_word(word: str) -> bool:
    current_stack = inspect.stack()

    # The first frame (index 0) is check_stack_for_word itself.
    # We iterate from the immediate caller (index 1) upwards.
    # We use enumerate to track the index so we can skip the code_context
    # of the immediate caller to avoid the false positive you identified.
    for i, frame_info in enumerate(current_stack):
        # Skip our own frame (index 0)
        if i == 0:
            continue
            
        function_name = frame_info.function
        filename = frame_info.filename
        # Check if the word is in the function name
        if word in function_name:
            # print(f"Found '{word}' in function name: '{function_name}'")
            return True
            
        if i > 1 and frame_info.code_context:
            for line in frame_info.code_context:
                if word in line:
                    # print(f"Found '{word}' in code context: '{line.strip()}'")
                    return True

    # print(f"'{word}' was not found in the stack trace.")
    return False

def _dump(tensor: torch.Tensor, name: str, layer: int, warm_up: bool):
    """Save a tensor to disk only if debugging is on."""
    from pathlib import Path
    import os
    
    rank = get_tensor_model_parallel_rank()
    if layer == 0 and not warm_up:
        debug_root = Path("/data/p00603624/debug_tensors")

        #  Ensure the directory exists (creates parents if needed)
        # `parents=True` = create any missing parent dirs (e.g. "./debug_tensors")
        # `exist_ok=True` = don’t raise an exception if it already exists
        debug_root.mkdir(parents=True, exist_ok=True)
        cpu_tensor = tensor.cpu()
        file_path = debug_root / f"rank_{rank}_{name}.pt"
        torch.save(cpu_tensor, file_path)

        if torch.is_floating_point(cpu_tensor):
            mean_val = cpu_tensor.mean().item()
            std_val  = cpu_tensor.std().item()
        else:
            mean_val = std_val = None
        print(
            f"[DEBUG] saved {name}: "
            f"path={file_path}, shape={cpu_tensor.shape}, "
            f"mean={mean_val if mean_val is not None else 'N/A'}, "
            f"std={std_val if std_val is not None else 'N/A'}"
        )

def _dump0(tensor: torch.Tensor, name: str, layer: int, warm_up: bool):
    """Save a tensor to disk only if debugging is on."""
    from pathlib import Path
    import os

    if layer == 0 and not warm_up and get_tensor_model_parallel_rank() == 0:
        debug_root = Path("/data/p00603624/debug_tensors")

        #  Ensure the directory exists (creates parents if needed)
        # `parents=True` = create any missing parent dirs (e.g. "./debug_tensors")
        # `exist_ok=True` = don’t raise an exception if it already exists
        debug_root.mkdir(parents=True, exist_ok=True)
        cpu_tensor = tensor.cpu()
        file_path = debug_root / f"{name}.pt"
        torch.save(cpu_tensor, file_path)

        if torch.is_floating_point(cpu_tensor):
            mean_val = cpu_tensor.mean().item()
            std_val  = cpu_tensor.std().item()
        else:
            mean_val = std_val = None
        print(
            f"[DEBUG] saved {name}: "
            f"path={file_path}, shape={cpu_tensor.shape}, "
            f"mean={mean_val if mean_val is not None else 'N/A'}, "
            f"std={std_val if std_val is not None else 'N/A'}"
        )

class LongcatFlashMLP(ParallelDeepseekMLP):
    pass

class LongcatFlashDecoderLayer(nn.Module):
    def __init__(
            self,
            config: PretrainedConfig,
            prefix: str,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.layer_name = f"{prefix}.self_attn.0.attn"
        self.hidden_size = config.hidden_size
        self.quant_symbol = quant_config is not None
        rope_theta = getattr(config, "rope_theta", 10000)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        self.self_attn = nn.ModuleList(
            [
                LongcatFlashMLA(
                    config=config,
                    hidden_size=self.hidden_size,
                    num_heads=config.num_attention_heads,
                    qk_nope_head_dim=config.qk_nope_head_dim,
                    qk_rope_head_dim=config.qk_rope_head_dim,
                    v_head_dim=config.v_head_dim,
                    q_lora_rank=config.q_lora_rank if hasattr(config, "q_lora_rank") else None,
                    kv_lora_rank=config.kv_lora_rank,
                    rope_theta=rope_theta,
                    rope_scaling=None,
                    max_position_embeddings=max_position_embeddings,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.self_attn.{i}",
                )
                for i in range(2)
            ]
        )
        self.input_layernorm = nn.ModuleList(
            [RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for i in range(2)]
        )
        self.post_attention_layernorm = nn.ModuleList(
            [RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for i in range(2)]
        )

        self.mlps = nn.ModuleList(
            [
                LongcatFlashMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.ffn_hidden_size,
                    hidden_act=config.hidden_act,
                    quant_config=quant_config,
                    prefix=f"{prefix}.mlps.{i}",
                )
                for i in range(2)
            ]
        )

        self.mlp = LongcatFlashMoE(
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
            residual: Optional[torch.Tensor],
            layer_id: Optional[int] = None,
    ) -> torch.Tensor:
        import traceback
        is_dummy_run = check_stack_for_word("dummy")
        # _dump(hidden_states, f"layer{layer_id}_input_hs", layer_id, is_dummy_run)
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.layer_name]
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm[0](hidden_states)
        else:
            # Adapt: adapt for w8a8 dynamic, do quant
            # Combines residual add and rmsnorm
            hidden_states, residual = self.input_layernorm[0](
                hidden_states, residual, quant_symbol=(not model_extra_config.operator_opt_config.use_mlaprolog and self.quant_symbol))
            # Adapt end.
        hidden_states = self.self_attn[0](
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        # _dump(hidden_states, f"layer{layer_id}_self_attn0_out", layer_id, is_dummy_run)
        hidden_states, residual = self.post_attention_layernorm[0](hidden_states, residual)
        # _dump(hidden_states, f"layer{layer_id}_post_ln0_out", layer_id, is_dummy_run)       # 已对齐
        # _dump0(residual,    f"layer{layer_id}_post_ln0_residual", layer_id, is_dummy_run)    # 已对齐
        moe_hidden_states = hidden_states.clone()

        # _dump(moe_hidden_states, f"layer{layer_id}_moe_input", layer_id, is_dummy_run)
        moe_hidden_states = self.mlp(moe_hidden_states, attn_metadata)
        if isinstance(moe_hidden_states, (tuple, list)):
            assert len(moe_hidden_states) == 2
            # 0 is the shared expert hidden_states, 1 is the routing expert hidden_states, add operation cannot be placed in the super kernel
            moe_hidden_states = moe_hidden_states[0] + moe_hidden_states[1]
        # _dump(moe_hidden_states, f"layer{layer_id}_moe_output", layer_id, is_dummy_run)

        hidden_states, residual = self.mlps[0](hidden_states, residual, attn_metadata)
        # _dump0(hidden_states, f"layer{layer_id}_mlp0_out", layer_id, is_dummy_run)

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm[1](hidden_states)
            # _dump0(hidden_states, f"layer{layer_id}_ln1_output", layer_id, is_dummy_run)
        else:
            # Adapt: adapt for w8a8 dynamic, do quant
            # Combines residual add and rmsnorm
            hidden_states, residual = self.input_layernorm[1](
                hidden_states, residual, quant_symbol=(not model_extra_config.operator_opt_config.use_mlaprolog and self.quant_symbol))
            # _dump0(hidden_states, f"layer{layer_id}_ln1_output", layer_id, is_dummy_run)
            # _dump0(residual,    f"layer{layer_id}_ln1_residual", layer_id, is_dummy_run)
        
        assert hidden_states.shape[0] > 0
        hidden_states = self.self_attn[1](
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        # _dump0(hidden_states, f"layer{layer_id}_self_attn1_out", layer_id, is_dummy_run)

        hidden_states, residual = self.post_attention_layernorm[1](hidden_states, residual)
        # _dump0(hidden_states, f"layer{layer_id}_post_ln1_out", layer_id, is_dummy_run)
        # _dump0(residual,    f"layer{layer_id}_post_ln1_residual", layer_id, is_dummy_run)

        hidden_states, residual = self.mlps[1](hidden_states, residual, attn_metadata)
        # _dump0(hidden_states, f"layer{layer_id}_mlp1_out", layer_id, is_dummy_run)

        hidden_states = moe_hidden_states + hidden_states
        # _dump0(hidden_states, f"layer{layer_id}_final_output", layer_id, is_dummy_run)

        return hidden_states, residual


class LongcatFlashModel(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):

        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.prefix = f"{prefix}.layers"
        self.postfix = ".self_attn.attn"
        self.tp_size = get_tensor_model_parallel_world_size()
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: LongcatFlashDecoderLayer(
                config,
                prefix,
                cache_config=cache_config,
                quant_config=quant_config,
            ),
            prefix=f"{prefix}.layers")

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids, reduce=1)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            intermediate_tensors: Optional[IntermediateTensors],
    ) -> Union[torch.Tensor, IntermediateTensors]:
        is_dummy_run = check_stack_for_word("dummy")
        if get_pp_group().is_first_rank:
            hidden_states = self.get_input_embeddings(input_ids)
            residual = None
            # _dump(input_ids, f"input_ids", 0, is_dummy_run)
            # _dump(hidden_states, f"input_embedding", 0, is_dummy_run)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(positions,
                                            hidden_states,
                                            kv_caches[i - self.start_layer] if kv_caches is not None else None,
                                            attn_metadata,
                                            residual,
                                            i)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)

        hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)

        return hidden_states


@support_torch_compile
class LongcatFlashForCausalLM(nn.Module):

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"]
    }
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = LongcatFlashModel(vllm_config=vllm_config, prefix="model")

        self.lm_head = ParallelLMHead(self.config.vocab_size,
                                      self.config.hidden_size,
                                      quant_config=self.quant_config,
									  parallel_lmhead=(model_extra_config.parall_config.dp_size > 1))
        self.logits_processor = LogitsProcessor(self.config.vocab_size,
                                                logits_as_input=True)
        self.sampler = Sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

        self.return_hidden_states = True

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor] = None,
            attn_metadata: Union[AttentionMetadata, dict] = None,
            selected_indices: Optional[torch.Tensor] = None,
            intermediate_tensors: Optional[IntermediateTensors] = None,
            inputs_embeds = None,
            **kwargs
    ) -> Optional[torch.Tensor]:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors)
        if attn_metadata is None:
            logits = self.compute_lmhead(hidden_states[-1:, ...], None)
        else:
            logits = self.compute_lmhead(hidden_states, selected_indices)

        if self.return_hidden_states:
            return hidden_states, logits
        else:
            return logits

    def compute_lmhead(
            self,
            hidden_states: torch.Tensor,
            selected_indices: Optional[torch.Tensor] = None,
            embedding_bias: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if selected_indices is not None:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            if hidden_states.shape[0] != selected_indices.shape[0]:
                hidden_states = hidden_states.index_select(0, selected_indices)

        # Get the logits for the next tokens.
        logits = self.lm_head(hidden_states, embedding_bias)
        return logits

    def compute_logits(
            self,
            hidden_states: torch.Tensor,
            sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)

        return logits

    def sample(
            self,
            logits: Optional[torch.Tensor],
            sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def make_empty_intermediate_tensors(
            self, batch_size: int, dtype: torch.dtype,
            device: torch.device) -> IntermediateTensors:
        return IntermediateTensors({
            "hidden_states":
                torch.zeros((batch_size, self.config.hidden_size),
                            dtype=dtype,
                            device=device),
            "residual":
                torch.zeros((batch_size, self.config.hidden_size),
                            dtype=dtype,
                            device=device),
        })


    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts)

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        total_count = 0
        loaded_count = 0
        for name, loaded_weight in weights:
            total_count += 1
            # rank0_log(f"to load weight name = {name}, shape = {loaded_weight.shape}, dtype = {loaded_weight.dtype}")
            if "rotary_emb.inv_freq" in name:
                rank0_log(f"[rotary_emb] failed to load weight name = {name}")
                continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    # rank0_log(f"[stacked_params_mapping] failed to load weight name = {name}, stacked weight name = {weight_name}")
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if (("mlp.experts." in name) and name not in params_dict):
                    # rank0_log(f"[mlp.experts] failed to load weight name = {name}")
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    # rank0_log(f"[bias0] failed to load weight name = {name}")
                    continue

                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    # rank0_log(f"[name not in params_dict0] failed to load weight name = {name}")
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                # rank0_log(f"[weight_loader1] success to load weight name = {name}")
                weight_loader(param, loaded_weight, shard_id)
                loaded_count += 1
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        # rank0_log(f"[name not in params_dict1] failed to load weight name = {name}")
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    # rank0_log(f"[weight_loader2] success to load weight name = {name}")
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    loaded_count += 1
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        # rank0_log(f"[bias] failed to load weight name = {name}")
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        # rank0_log(f"[name not in params_dict2] failed to load weight name = {name}")
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    # rank0_log(f"[weight_loader3] success to load weight name = {name}")
                    weight_loader(param, loaded_weight)
                    loaded_count += 1
            loaded_params.add(name)
        rank0_log(f"load weight count = {loaded_count}, total_count = {total_count}")
        return loaded_params

    def should_use_eager_mode(self, *args, **kwargs):
        # attn_metadata = kwargs.get("attn_metadata", None)
        # if not attn_metadata:
        #     return True

        # if isinstance(attn_metadata, dict):
        #     attn_metadata = attn_metadata[self.model.layers[self.model.start_layer].layer_name]

        # if attn_metadata.prefill:
        #     return True

        # return False
        return True
