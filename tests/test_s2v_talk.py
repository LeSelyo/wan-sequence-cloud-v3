import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import s2v_talk as st  # noqa: E402


def references_exist(graph):
    for node_id, node in graph.items():
        for value in node["inputs"].values():
            if isinstance(value, list):
                assert value[0] in graph, (node_id, value)


def test_frames_follow_the_4k_plus_1_rule_of_wan():
    assert [st.frames_for(s) for s in (1, 2, 3, 5)] == [17, 33, 49, 81]
    assert all((st.frames_for(s / 10) - 1) % 4 == 0 for s in range(5, 80))


def test_graph_without_lora_is_the_reference_setup_and_every_link_exists():
    graph = st.build_graph("p.png", "a.wav", "x", seed=7, lora="none", seconds=3)
    references_exist(graph)
    assert not any(n["class_type"] == "LoraLoaderModelOnly" for n in graph.values())
    sampler = graph["13"]["inputs"]
    assert (sampler["steps"], sampler["cfg"], sampler["seed"]) == (20, 4.5, 7) and graph["12"]["inputs"]["model"] == ["1", 0]
    assert graph["10"]["inputs"]["length"] == 49 and (graph["10"]["inputs"]["width"], graph["10"]["inputs"]["height"]) == (480, 832)
    assert graph["15"]["inputs"]["audio"] == ["5", 0] and graph["16"]["inputs"]["video"] == ["15", 0]


def test_graph_with_a_distillation_lora_is_4_steps_without_cfg_and_the_lora_feeds_the_sampler():
    for lora in ("animate_lightx", "i2v_low"):
        graph = st.build_graph("p.png", "a.wav", "x", lora=lora)
        references_exist(graph)
        assert graph["20"]["inputs"]["lora_name"] == st.LORAS[lora][0][0][0] and graph["12"]["inputs"]["model"] == ["20", 0]
        assert (graph["13"]["inputs"]["steps"], graph["13"]["inputs"]["cfg"]) == (4, 1.0)
    assert st.build_graph("p.png", "a.wav", "x", lora="i2v_low", steps=6, cfg=1.5)["13"]["inputs"]["steps"] == 6


def test_several_loras_are_chained_and_the_sampler_is_fed_by_the_last_one():
    graph = st.build_graph("p.png", "a.wav", "x", lora="i2v_low_real")
    references_exist(graph)
    assert graph["20"]["inputs"]["model"] == ["1", 0] and graph["21"]["inputs"]["model"] == ["20", 0] and graph["12"]["inputs"]["model"] == ["21", 0]
    assert graph["21"]["inputs"]["lora_name"] == "Instareal_low.safetensors" and graph["21"]["inputs"]["strength_model"] == 0.8
    soft = st.build_graph("p.png", "a.wav", "x", lora="i2v_low_soft")
    assert (soft["13"]["inputs"]["steps"], soft["13"]["inputs"]["cfg"], soft["20"]["inputs"]["strength_model"]) == (8, 1.5, 0.7)
    big = st.build_graph("p.png", "a.wav", "x", size=(576, 1024))
    assert (big["10"]["inputs"]["width"], big["10"]["inputs"]["height"]) == (576, 1024)
