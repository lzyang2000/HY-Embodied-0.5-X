"""HY-Embodied-0.5-X: an embodied multimodal foundation model.

Public library surface:
    from hy_embodied.inference import HyEmbodiedPipeline
    from hy_embodied.training import SFTConfig, run_sft
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hy_embodied")
except PackageNotFoundError:  # running from source tree without install
    __version__ = "0.5.0-X"

__all__ = ["__version__"]
