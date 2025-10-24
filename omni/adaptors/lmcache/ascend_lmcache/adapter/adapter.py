# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import importlib
import sys

def replace_module_use_ascend_lmcache(old_module_name: str):
    new_module_name = old_module_name.replace("lmcache", "ascend_lmcache")
    if old_module_name in sys.modules:
        del sys.modules[old_module_name]
    sys.modules[old_module_name] = importlib.import_module(new_module_name)

_default_ops = ("lmcache.c_ops", "lmcache.integration.vllm.vllm_adapter")

for _ops in _default_ops:
    replace_module_use_ascend_lmcache(_ops)

from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl
from ascend_lmcache.integration.vllm.vllm_v1_adapter import (
    start_load_kv,
    __init__,
    get_load_kv_failure_reqs,
)

LMCacheConnectorV1Impl.__init__ = __init__
LMCacheConnectorV1Impl.start_load_kv = start_load_kv
LMCacheConnectorV1Impl.get_load_kv_failure_reqs = get_load_kv_failure_reqs

from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import LMCacheConnectorV1
from ascend_lmcache.integration.vllm.lmcache_connector import get_load_kv_failure_reqs

LMCacheConnectorV1.get_load_kv_failure_reqs = get_load_kv_failure_reqs