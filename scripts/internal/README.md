# scripts/internal/

Developer-only tooling that is **not** part of the published workflow but
kept in tree for reproducibility. Nothing here is imported by the
`hy_embodied` package and nothing under this directory runs during training
or inference.

## `build_data_demo.py`

Rebuilds `data_examples/data_demo.jsonl` and the companion
`data_examples/images/` tree from our internal training corpus.

It reads category-specific JSONL snapshots placed under `data_examples/*.jsonl`
(not shipped with the release), keeps only samples whose image paths are
resolvable on the current machine, copies those images into
`data_examples/images/<category>/`, and rewrites the `image` fields to
repo-relative paths so the demo is fully self-contained.

Run from the repo root:

```bash
python scripts/internal/build_data_demo.py
```

External contributors will not have the source JSONLs, so this script is
effectively a no-op for them — they should craft `data_examples/data_demo.jsonl`
by hand instead.
