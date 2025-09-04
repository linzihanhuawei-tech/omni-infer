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
"""Inference-only DeepseekV3 model."""
from typing import Optional
import torch, torch_npu
from torch import nn
from transformers import PretrainedConfig
import torchair as tng
torch._logging.set_logs(recompiles=True)
# vllm adaptor
from vllm.platforms import current_platform
from vllm.config import QuantizationConfig
from vllm.attention import AttentionMetadata
from vllm.distributed import (
    get_ep_group,
    get_dp_group,
    get_world_group,
    get_tensor_model_parallel_rank
)
from vllm.model_executor.layers.linear import ReplicatedLinear

from omni.models.longcat.longcat_fused_moe import FusedMoE
from omni.models.common.config.model_config import model_extra_config
from omni.adaptors.vllm.patches.model_patch import get_attr_by_names


"""NPU Stream Switch Names"""
STREAM_SHARED_EXPERT = 'stream_shared_expert'
SEQ_SPLIT_LENGTH = 4096
def rank0_log(msg):
    if get_tensor_model_parallel_rank() == 0:
        print(msg)

def get_layer_id_from_prefix(prefix: str):
    import re
    match = re.search(r'\d+', prefix)
    return int(match.group())

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

class LongcatFlashTopkRouter(nn.Module):
    def __init__(
        self, 
        config, 
        zero_expert_num: int = 0,
        router_expert_num: int = 256,
        prefix: str = "", 
        rounter_params_dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.config = config
        self.top_k = config.moe_topk
        self.n_routed_experts = router_expert_num + zero_expert_num
        self.routed_scaling_factor = config.routed_scaling_factor

        self.classifier = ReplicatedLinear(
            config.hidden_size,
            self.n_routed_experts,
            bias=config.router_bias,
            params_dtype=rounter_params_dtype,
            quant_config=None,
            prefix=f"{prefix}.classifier",
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros((self.n_routed_experts), dtype=rounter_params_dtype)
        )

    # @torch.no_grad()
    # def get_topk_indices(self, scores):
    #     scores_for_choice = scores.view(-1, self.n_routed_experts) + self.e_score_correction_bias.unsqueeze(0)
    #     topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
    #     return topk_indices

    # def forward(self, hidden_states):
    #     hidden_states = hidden_states.view(-1, self.config.hidden_size)
    #     router_logits, _ = self.classifier(hidden_states.type(torch.float32))
    #     scores = router_logits.softmax(dim=-1)
    #     topk_indices = self.get_topk_indices(scores)
    #     topk_weights = scores.gather(1, topk_indices)
    #     if self.norm_topk_prob:
    #         denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
    #         topk_weights /= denominator
    #     topk_weights = topk_weights * self.routed_scaling_factor
    #     return topk_indices, topk_weights
    def forward(self, hidden_states):
        return self.classifier(hidden_states)
    
    def get_topk_indices_original(self, router_logits):
        # npu_moe_gating_top_k算子有问题，只支持256和384专家
        # return torch_npu.npu_moe_gating_top_k(
        #     router_logits.float(),
        #     k=self.top_k,  # topk is currently 8
        #     bias=self.e_score_correction_bias,  # float32
        #     k_group=1,  # fix: 4
        #     group_count=self.n_routed_experts,  # fix 8
        #     group_select_mode=1,  # 0: maximum in group; 1: topk2.sum(fix)
        #     renorm=0,  # 0: softmax->topk(fix); 1: topk->softmax
        #     norm_type=1,  # 0: softmax; 1: sigmoid(fix)
        #     routed_scaling_factor=self.routed_scaling_factor,
        #     eps=float(1e-20)
        # )
        router_logits = router_logits.view(-1, self.n_routed_experts) + self.e_score_correction_bias.unsqueeze(0)
        topk_weights, topk_ids, row_idx = torch_npu.npu_moe_gating_top_k_softmax(router_logits, k=self.top_k)

        topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_weights, topk_ids, row_idx

    def get_topk_indices(self, classify_score, layer):
        is_dummy = check_stack_for_word("dummy")

        n_routed_experts = classify_score.shape[-1]
        scores = classify_score.softmax(dim=-1)
        scores_for_choice = scores.view(-1, n_routed_experts) + self.e_score_correction_bias.unsqueeze(0) # 对齐
        # _dump(scores_for_choice, "score_for_choice", layer, is_dummy)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1] # 对齐
        topk_weights = scores_for_choice.gather(1, topk_indices)
        topk_weights = topk_weights.to(torch.float32) * self.routed_scaling_factor
        return topk_weights.to(torch.float32), topk_indices.to(torch.int32)

class LongcatFlashMoE(nn.Module):

    def __init__(
            self,
            config: PretrainedConfig,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
    ):
        super().__init__()
        self.prefix = prefix
        self.ep_size = get_ep_group().world_size
        self.routed_scaling_factor = config.routed_scaling_factor

        n_routed_experts_names = ['num_routed_experts', 'n_routed_experts']
        self.n_routed_experts = get_attr_by_names(config, n_routed_experts_names, 256)
        self.zero_expert_num = config.zero_expert_num
        self.redundancy_shared_expert_num = model_extra_config.parall_config.redundancy_shared_expert_num
        self.quant_symbol = quant_config is not None
        self.is_init_gate = False

        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}. "
                             "Only silu is supported for now.")

        router_params_dtype = torch.float32
        self.router = LongcatFlashTopkRouter(config=config,
                                             zero_expert_num=self.zero_expert_num,
                                             router_expert_num=self.n_routed_experts,
                                             rounter_params_dtype=router_params_dtype,
                                             prefix=f"{prefix}.router")

        self.top_k = config.moe_topk
        self.renormalize = getattr(config, "norm_topk_prob", True)
        self.global_rank = get_world_group().rank_in_group

        moe_prefix = f"{prefix}.experts"
        # omni placement for redundancy route experts
        self.experts = FusedMoE(
            num_experts=self.n_routed_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.expert_ffn_hidden_size,
            reduce_results=False,
            quant_config=quant_config,
            prefix=moe_prefix,
        )
        self.local_expert_num = self.experts.w13_weight.shape[0]
        if self.quant_symbol:
            self.in_scale_2 = torch.ones((self.local_expert_num, config.expert_ffn_hidden_size), dtype=torch.float32, device=current_platform.device_type)
            # call the mark_static to reduce memory usage
            torch._dynamo.mark_static(self.in_scale_2)

        self.tuning_config = None
        if model_extra_config.operator_opt_config.gmm_nz:
            self.tuning_config = model_extra_config.operator_opt_config.decode_gear_list[:1]

    def forward(self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata) -> torch.Tensor:
        if not self.is_init_gate:
            self.router.classifier.weight.data = torch_npu.npu_format_cast(self.router.classifier.weight.data, 2)
            self.is_init_gate = True
        if attn_metadata is None or attn_metadata.prefill is not None:
            return self._forward_prefill_norm(hidden_states, attn_metadata)
        else:
            return self._forward_decode_norm(hidden_states, attn_metadata)

    def compute_zero_experts(self, hidden_states, topk_weights, topk_ids):
        zero_expert_mask = topk_ids >= self.n_routed_experts
        normal_expert_mask = topk_ids < self.n_routed_experts

        zero_expert_weights = topk_weights.clone()
        zero_expert_weights[normal_expert_mask] = 0
        total_weights = zero_expert_weights.sum(dim=-1, keepdim=True)   # [T, 1]
        # zero_expert_output = hidden_states * total_weights
        zero_expert_output = hidden_states * total_weights              # [T, H]
        zero_expert_output = zero_expert_output.to(hidden_states.dtype)
        topk_ids[zero_expert_mask] = 0 # 都用0号专家替代
        topk_weights[zero_expert_mask] = 0.0
        return zero_expert_output, topk_ids, topk_weights

    def _forward_prefill_norm(self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata) -> torch.Tensor:
        # is_dummy = check_stack_for_word("dummy")
        layer = get_layer_id_from_prefix(self.prefix)

        router_logits, _ = self.router.forward(hidden_states.float())
        # _dump0(router_logits, "router_logits", layer, is_dummy)
        topk_weights, topk_ids = self.router.get_topk_indices(router_logits, layer)
        # _dump0(topk_ids, "topk_ids", layer, is_dummy)
        # _dump0(topk_weights, "topk_weights", layer, is_dummy)
        zero_expert_output, topk_ids, topk_weights = self.compute_zero_experts(hidden_states, topk_weights, topk_ids)
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            pertoken_scale=None,
            attn_metadata=attn_metadata
        )
        if isinstance(final_hidden_states, tuple):
            gathered_tokens = final_hidden_states[1]
            expanded_row_idx = final_hidden_states[3]
            final_hidden_states = torch_npu.npu_moe_finalize_routing(
                gathered_tokens,
                skip1=zero_expert_output,
                skip2=None,
                bias=None,
                scales=topk_weights.to(gathered_tokens.dtype),
                expanded_src_to_dst_row=expanded_row_idx,
                export_for_source_row=None,
                drop_pad_mode=2
            )
        else:
            final_hidden_states = final_hidden_states + zero_expert_output

        return final_hidden_states

    def _forward_decode_norm(self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata) -> torch.Tensor:
        if model_extra_config.operator_opt_config.use_super_kernel:
            with tng.scope.super_kernel(self.prefix, 'stream-fusion=1'):
                return self._forward_decode_dispatch_combine(hidden_states, attn_metadata)
        else:
            return self._forward_decode_dispatch_combine(hidden_states, attn_metadata)

    def _forward_decode_dispatch_combine(self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata) -> torch.Tensor:
        router_logits, _ = self.router.forward(hidden_states.float())
        layer = get_layer_id_from_prefix(self.prefix)
        topk_weights, topk_ids = self.router.get_topk_indices(router_logits, layer)
        zero_expert_output, topk_ids, topk_weights = self.compute_zero_experts(hidden_states, topk_weights, topk_ids)

        mc2_mask = attn_metadata.decode.mc2_mask if attn_metadata is not None and attn_metadata.decode is not None else None
        layer = self.experts

        max_num_deployed_expert = self.local_expert_num * get_dp_group().world_size
        act_dtype = hidden_states.dtype
        shared_expert_rank_num = 0
        kwargs = {
            "x": hidden_states,
            "expert_ids": topk_ids,  # [n*topk]
            "expert_shard_type": 0,  # Set it to 0 for now
            "shared_expert_rank_num": shared_expert_rank_num,  # 32
            "moe_expert_num": max_num_deployed_expert, #ENABLE_OMNI_PLANNER, 0 redundancy 256, 1 redundancy expert 320
            "global_bs": 0,  # 0 Default (all); all tokens can be set
        }

        experts_tp_size = layer.tp_size
        world_size = get_world_group().world_size
        # In fact, what we get is the die number, and the ep group is not adapted by default.
        # The default ep group is experts_num/die_num.
        global_rank = get_world_group().rank_in_group
        all_to_all_group_size = world_size // experts_tp_size

        kwargs.update({
            "scales": None,  # Quantization coefficient
            "quant_mode": layer.quant_mode,  # 0: Non-quantization; 1: Static quantization; 2: Dynamic quantization
            "group_ep": layer.moe_all_to_all_group_name,  # Unlike torch, it is obtained by name.
            "ep_world_size": all_to_all_group_size,
            "ep_rank_id": global_rank // experts_tp_size,
            "group_tp": layer.moe_rs_group_name,
            "tp_world_size": experts_tp_size,
            "tp_rank_id": global_rank % experts_tp_size,
            "x_active_mask": mc2_mask,
        })

        output = torch_npu.npu_moe_distribute_dispatch_v2(**kwargs)
        expand_x, dynamic_scale, expand_idx, expert_token_nums, ep_recv_counts = output[0:5]

        group_list = expert_token_nums.to(torch.int64)

        # cal experts
        weight1_3 = self.experts.w13_weight
        weight2 = self.experts.w2_weight

        # bf16
        gate_up_proj = torch_npu.npu_grouped_matmul([expand_x], [weight1_3], bias=None, group_list=group_list,
                                                split_item=3, group_type=0, group_list_type=1)[0]
    
        gate_up_proj = torch_npu.npu_swiglu(gate_up_proj)

        hidden_states_experts = torch_npu.npu_grouped_matmul([gate_up_proj], [weight2],bias=None,
                                        group_list=group_list, split_item=3, output_dtype=act_dtype,
                                        group_type=0, group_list_type=1)[0]

        # moeCombine
        kwargs = {
            "expand_x": hidden_states_experts,
            "expert_ids": topk_ids,  # [n*topk]
            "assist_info_for_combine": expand_idx,
            "expert_scales": topk_weights.to(torch.float32),  # weight [n*topk]
            "expert_shard_type": 0,
            "shared_expert_rank_num": shared_expert_rank_num,
            "moe_expert_num":  max_num_deployed_expert, #ENABLE_OMNI_PLANNER, 0 redundancy 256, 1 redundancy expert 320
            "global_bs": 0,  # 0 Default (all); all tokens can be set
        }
        tp_recv_counts = output[5]
        stage3_kwargs = {
            "ep_send_counts": ep_recv_counts,  # dispatch's send_counts
            "group_ep": layer.moe_all_to_all_group_name,  # Unlike torch, it is obtained by name.
            "ep_world_size": all_to_all_group_size,
            "ep_rank_id": global_rank // experts_tp_size,
            "tp_send_counts": tp_recv_counts,
            "group_tp": layer.moe_rs_group_name,
            "tp_world_size": experts_tp_size,
            "tp_rank_id": global_rank % experts_tp_size,
            "x_active_mask": mc2_mask,
        }
        kwargs.update(stage3_kwargs)


        hidden_states_route = torch_npu.npu_moe_distribute_combine_v2(**kwargs)

        final_hidden_states = (hidden_states_route, zero_expert_output)
        return final_hidden_states
