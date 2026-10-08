from scripts import comfy_edit as ce


def _references(graph):
    for node_id, node in graph.items():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                yield node_id, value[0]


def test_workflow_has_no_dangling_reference_and_the_expected_chain():
    graph = ce.build_workflow("add people", "in.png", seed=7)
    for node_id, target in _references(graph):
        assert target in graph, f"node {node_id} points to a missing node {target}"
    classes = {n["class_type"] for n in graph.values()}
    assert {"UNETLoader", "LoraLoaderModelOnly", "ModelSamplingAuraFlow", "CFGNorm", "CLIPLoader", "VAELoader", "LoadImage", "ImageScaleToTotalPixels", "TextEncodeQwenImageEditPlus",
            "FluxKontextMultiReferenceLatentMethod", "VAEEncode", "KSampler", "VAEDecode", "SaveImage"} <= classes
    assert graph["14"]["inputs"]["seed"] == 7 and graph["14"]["inputs"]["steps"] == 4 and graph["14"]["inputs"]["cfg"] == 1.0
    assert graph["9"]["inputs"]["prompt"] == "add people" and graph["10"]["inputs"]["prompt"] == ""
    assert graph["5"]["inputs"]["type"] == "qwen_image"
    assert graph["14"]["inputs"]["latent_image"] == ["13", 0]  # the edit starts from the latent of the input image, so its size and composition are kept


def test_without_lightning_the_lora_is_left_out_and_the_model_goes_straight_to_the_sampler_chain():
    graph = ce.build_workflow("x", "in.png", lightning=False, steps=40, cfg=4.0)
    assert "2" not in graph and graph["3"]["inputs"]["model"] == ["1", 0]
    assert graph["14"]["inputs"]["steps"] == 40 and graph["14"]["inputs"]["cfg"] == 4.0


def test_constant_instructions_are_place_agnostic():
    forbidden = ("street", "train", "cafe", "bird", "rue", "wagon")
    for text in [ce.CROWD, *ce.LIGHT.values()]:
        assert not any(word in text.lower() for word in forbidden)
    assert set(ce.LIGHT) == {"dawn", "midday", "sunset", "night"}


def test_sage_attention_is_detected_from_the_arguments_comfyui_reports(monkeypatch):
    import httpx

    class Reply:
        def __init__(self, argv):
            self.argv = argv

        def json(self):
            return {"system": {"argv": self.argv}} if self.argv is not None else {"system": {}}

    for argv, expected in ((["main.py", "--use-sage-attention", "--novram"], True), (["main.py", "--novram"], False), (None, None)):
        monkeypatch.setattr(httpx, "get", lambda *a, _argv=argv, **k: Reply(_argv))
        assert ce.sage_attention_on() is expected
    monkeypatch.setattr(httpx, "get", lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    assert ce.sage_attention_on() is None  # nothing answers: cannot tell, the caller decides


def test_run_edit_refuses_to_run_with_sage_on(monkeypatch, tmp_path):
    import pytest
    monkeypatch.setattr(ce, "sage_attention_on", lambda *a, **k: True)
    with pytest.raises(RuntimeError, match="BLACK"):
        ce.run_edit(tmp_path / "in.png", tmp_path / "out.png", "x")
