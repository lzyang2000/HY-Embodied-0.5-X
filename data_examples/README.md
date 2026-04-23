# data_examples/

A tiny, **self-contained** training corpus used to smoke-test the data
pipeline and to document the expected JSONL schema. Everything under this
directory is shipped with the repo (≈ 1 MB of images) so the default
training config works out of the box, with no external data or mounted
network drives required.

```
data_examples/
├── data_demo.jsonl     # 14 samples across 6 capabilities
└── images/
    ├── affordance/     # manipulation affordance frames (4 frames per sample)
    ├── refspatial/     # spatial-grounding (multi-turn)
    ├── robot_traj/     # robot trajectory / spatial QA
    └── trajectory/     # end-effector trajectory prediction
```

Two of the 14 samples (`planning`) are pure text; two more reference images
shipped in the repo under `assets/` (result figures). The remaining ten
reference the bundled `images/<category>/` copies.

## JSONL schema

Every non-empty line is one sample:

```json
{
  "id": "optional-sample-id",
  "type": "optional-label",
  "_category": "optional-tag-used-only-for-this-demo",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "data_examples/images/xxx.jpg"},
        {"type": "text",  "text": "Question /think or /no_think"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "<think>\n...\n</think>\n<answer>\n...\n</answer>"}
      ]
    }
  ]
}
```

See [`docs/data_format.md`](../docs/data_format.md) for the full schema
(think / no_think modes, multi-image, video, coordinate conventions, and
the assistant-only loss-masking contract).

## Capabilities covered in the demo

| `_category`        | # samples | Content | Images |
|--------------------|-----------|---------|--------|
| `self_contained`   | 4         | Mixed Q&A using repo figures from `assets/` | 3 |
| `planning`         | 2         | Long-horizon task planning (pure text, CoT) | 0 |
| `trajectory`       | 2         | End-effector trajectory prediction on (0–1000)² | 2 |
| `affordance`       | 2         | Affordance reasoning over multi-frame clips | 8 |
| `refspatial`       | 2         | Spatial-grounding, multi-turn Q&A on one image | 2 |
| `robot_traj`       | 2         | Robot trajectory + spatial reasoning, mixed CoT / direct | 2 |

Not included (the underlying images are not redistributable under this
repo's license): `bbox / point grounding`, `UMI / xTrainer / xPerience`
manipulation QA, `referring_intention`. Users who have access to those data
sources can follow the exact same schema to assemble their own JSONL.

## Building your own training mixture

1. Follow the schema above, one sample per JSONL line.
2. Put all data files somewhere (inside `data_examples/` or elsewhere) and
   list them under `data.train_data_paths` in a training config.
3. Optionally add entries under `data.train_data_sampling_ratios` to
   downsample (< 1), upsample (> 1), or skip (= 0) individual files.
4. See [`configs/sft/example_small.yaml`](../configs/sft/example_small.yaml)
   — the release-recipe training / optimizer defaults are already filled
   in there, so in practice you only need to edit the two fields above.
