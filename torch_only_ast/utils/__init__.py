import os

__all__ = [
    "cache_dir",
    "model_cache_dir",
]


_home_dir = os.path.expanduser("~")
cache_dir = os.getenv("TORCHONLYAST_CACHE_DIR") or os.path.join(
    _home_dir, ".cache", "torch_only_ast"
)
model_cache_dir = os.path.join(cache_dir, "models")
