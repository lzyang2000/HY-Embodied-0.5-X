"""Backward-compatible shim for ``python inference.py``.

New code should prefer::

    python -m hy_embodied.cli.infer --model ckpts/HY-Embodied-0.5-X --image assets/demo.jpg

or the library API::

    from hy_embodied.inference import HyEmbodiedPipeline
    pipe = HyEmbodiedPipeline.from_pretrained("ckpts/HY-Embodied-0.5-X")
    pipe.generate("Describe this image.", image="assets/demo.jpg")

This file exists so that older documentation commands such as
``python inference.py --image assets/demo.jpg --prompt "..."`` keep working.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from a source checkout without `pip install -e .`
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hy_embodied.cli.infer import main  # noqa: E402


if __name__ == "__main__":
    main()
