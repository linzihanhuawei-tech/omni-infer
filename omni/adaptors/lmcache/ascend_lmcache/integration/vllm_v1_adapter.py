# Standard
from typing import TYPE_CHECKING, Optional

import torch

from ascend_lmcache.integration.vllm.vllm_adapter import init_lmcache_engine, init_p2p_engine
from ascend_lmcache.integration.vllm.vllm_v1_async_adapter import LMCacheConnectorMetadata
from lmcache.integration.vllm.utils import (
    ENGINE_NAME,
    lmcache_get_config,
)

from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.compute.blend import LMCBlenderBuilder
from lmcache.v1.lookup_client import LookupClientFactory

from lmcache.integration.vllm.vllm_v1_adapter import LoadSpec, RequestTracker
# Third Party
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole
)

if TYPE_CHECKING:
    # Third Party
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.request import Request

logger = init_logger(__name__)


def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        parent: KVConnectorBase_V1,
):
    self._parent = parent
    self.kv_role = vllm_config.kv_transfer_config.kv_role
    is_tp = vllm_config.parallel_config.tensor_parallel_size > 1

    config = lmcache_get_config()
    self.layerwise_retrievers = []
    if role == KVConnectorRole.SCHEDULER:
        # Create lookup client using factory
        self.lookup_client = LookupClientFactory.create_lookup_client(
            role, is_tp, vllm_config
        )
        self._requests_in_step: dict[str, Request] = {}
    else:
        self.lmcache_engine = init_lmcache_engine(
            vllm_config.model_config,
            vllm_config.parallel_config,
            vllm_config.cache_config,
            vllm_config.scheduler_config,
        )

        self.use_layerwise = config.use_layerwise
        self.enable_blending = config.enable_blending

        if self.enable_blending:
            self.blender = LMCBlenderBuilder.get_or_create(
                ENGINE_NAME,
                self.lmcache_engine,
                self.lmcache_engine.gpu_connector,
            )

        # Create lookup server using factory
        assert self.lmcache_engine is not None
        self.lookup_server = LookupClientFactory.create_lookup_server(
            self.lmcache_engine, role, is_tp, vllm_config
        )

    self.kv_caches: dict[str, torch.Tensor] = {}

    self._block_size = vllm_config.cache_config.block_size

    # request_id -> (vllm cached tokes, lmcache cached tokens)
    self.load_specs: dict[str, LoadSpec] = {}

    self.kv_cache_manager: Optional[KVCacheManager] = None

    # request_id -> full_token_ids
    self._request_trackers: dict[str, RequestTracker] = {}

    # Whether to discard partial chunks
    self._discard_partial_chunks = (
        vllm_config.kv_transfer_config.get_from_extra_config(
            "discard_partial_chunks", False
        )
    )

    self._lmcache_chunk_size = config.chunk_size

    self.skip_last_n_tokens = vllm_config.kv_transfer_config.get_from_extra_config(
        "skip_last_n_tokens", 0
    )

    self.num_layers = vllm_config.model_config.get_num_layers(
        vllm_config.parallel_config
    )
    self.current_layer = 0

    # kv loading failure reqs
    self._load_kv_failure_reqs: set[str] = set()


@_lmcache_nvtx_annotate
def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
    """Start loading the KV cache from the connector buffer to vLLM's
    paged KV buffer.

    Args:
        forward_context (ForwardContext): the forward context.
        **kwargs: additional arguments for the load operation

    Note:
        The number of elements in kv_caches and layer_names should be
        the same.
    """
    self.current_layer = 0

    if len(self.kv_caches) == 0:
        self._init_kv_caches_from_forward_context(forward_context)

    metadata = self._parent._get_connector_metadata()
    assert isinstance(metadata, LMCacheConnectorMetadata)

    assert len(self.kv_caches) > 0
    kvcaches = list(self.kv_caches.values())

    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        logger.warning("In connector.start_load_kv, but the attn_metadata is None")
        return

    assert self.lmcache_engine is not None

    for idx, request in enumerate(metadata.requests):
        if request.load_spec is None:
            continue

    self.layerwise_retrievers = []
    for idx, request in enumerate(metadata.requests):
        if request.load_spec is None:
            continue

        tokens = request.token_ids
        # TODO: have a pre-allocated buffer to hold the slot_mappings
        slot_mapping = request.slot_mapping.cuda()
        assert len(tokens) == len(slot_mapping)

        token_mask = torch.ones_like(tokens, dtype=torch.bool)
        masked_token_count = (
                request.load_spec.vllm_cached_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
        )
        token_mask[:masked_token_count] = False

        lmcache_cached_tokens = request.load_spec.lmcache_cached_tokens
        if self.use_layerwise:
            sync = True
            # NOTE(Jiayi): Perform blending before layerwise prefix caching
            if self.enable_blending:
                # TODO(Jiayi): Need to make prefix caching and blending compatible
                self.blender.blend(
                    tokens[:lmcache_cached_tokens],
                    token_mask[:lmcache_cached_tokens],
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping[:lmcache_cached_tokens],
                )
            else:
                layerwise_retriever = self.lmcache_engine.retrieve_layer(
                    tokens[:lmcache_cached_tokens],
                    token_mask[:lmcache_cached_tokens],
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping[:lmcache_cached_tokens],
                    sync=sync,
                )
                # NOTE: retrieve for two layers at the first layer
                next(layerwise_retriever)
                next(layerwise_retriever)
                self.layerwise_retrievers.append(layerwise_retriever)
        else:
            ret_token_mask = self.lmcache_engine.retrieve(
                tokens[:lmcache_cached_tokens],
                token_mask[:lmcache_cached_tokens],
                kvcaches=kvcaches,
                slot_mapping=slot_mapping[:lmcache_cached_tokens],
            )

            # Check the result
            num_retrieved_tokens = ret_token_mask.sum().item()
            num_expected_tokens = (
                    lmcache_cached_tokens - request.load_spec.vllm_cached_tokens
            )
            if num_retrieved_tokens < num_expected_tokens:
                logger.error(
                    "The number of retrieved tokens is less than the "
                    "expected number of tokens! This should not happen!"
                )
                logger.error(
                    "Num retrieved tokens: %d, num expected tokens: %d",
                    num_retrieved_tokens,
                    num_expected_tokens,
                )
                self._load_kv_failure_reqs.add(request.req_id)


def get_load_kv_failure_reqs(self) -> Optional[set[str]]:
    """
    Return kv loading failure request
    """
    all_load_kv_failure_reqs = set(self._load_kv_failure_reqs)
    self._load_kv_failure_reqs.clear()
    return all_load_kv_failure_reqs