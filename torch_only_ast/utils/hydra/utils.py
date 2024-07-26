from typing import Any

import hydra


def instantiate(config: Any, *args, **kwargs) -> Any:
    """Wrapper function of ``hydra.utils.instantiate``."""
    return hydra.utils.instantiate(config, *args, **kwargs)
