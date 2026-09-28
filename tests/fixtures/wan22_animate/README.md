# Wan 2.2 Animate template fixture

`video_wan2_2_14B_animate.json` is fetched verbatim from
`Comfy-Org/workflow_templates`, commit `9b912856b25a8564632b20857aa6353fbf78eb5a`
(the same commit pinned in the Dockerfile's `WORKFLOW_TEMPLATES_ANIMATE_COMMIT`
build arg), path `templates/video_wan2_2_14B_animate.json`.

sha256: `69ba18fffbc78cc59f35fc586f58575fa16424e6c112768aaae470d12867e182`

Used offline by `tests/test_wan_animate_subgraphs.py` to regression-test
`expand_subgraphs()` against the template's real structure: 3 chained
subgraph instances (`Video Sampling and output` -> `Video Extend` -> `Video
Extend`), where each later instance's `image1`/`video_frame_offset` inputs
are wired directly from the previous instance's own subgraph outputs. This
is the exact shape that used to raise `"a subgraph input connected from
another subgraph is unsupported"`.
