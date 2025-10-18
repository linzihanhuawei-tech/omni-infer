# Standard
from typing import Optional


def get_load_kv_failure_reqs(self) -> Optional[set[str]]:
    return self._lmcache_engine.get_load_kv_failure_reqs()