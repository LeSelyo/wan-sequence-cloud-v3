import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from app.comfy import bind_workflow, inject_loras, load_template
from scripts import prepare_workflows


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = PROJECT_ROOT / "tests/fixtures/workflow_templates_1b3bdd46"
PINNED_COMMIT = "1b3bdd46c945d54d893a3b43692d5963608fb7d4"
WORKFLOWS = {
    "wan22_t2v": "video_wan2_2_14B_t2v.json",
    "wan22_i2v": "video_wan2_2_14B_i2v.json",
    "wan22_flf2v": "video_wan2_2_14B_flf2v.json",
}
PINNED_SHA256 = {
    "video_wan2_2_14B_t2v.json": "c07feedfb87de638cb34bd517d916fc7afa7786be453e433a7bb49833a0c2801",
    "video_wan2_2_14B_i2v.json": "455337c85e3fb0c7da9b2e3e6408f02f4be3615ec9b77c0f18c4262b931dd650",
    "video_wan2_2_14B_flf2v.json": "9fb579e07caff9081c14a4c0e3b983e210aa7d976f83f1c2758d2ad6ed949fdf",
}
PINNED_CANONICAL_SHA256 = {
    "video_wan2_2_14B_t2v.json": "548bc9ddd65f7629a14eea5fc5d0cf7c68c6347af4f6fb3a879b0d04cdccee44",
    "video_wan2_2_14B_i2v.json": "d78af6d869998f23da938e3b142c9de66e0ca4c9b955c0b5e5fa4cfe938bc88c",
    "video_wan2_2_14B_flf2v.json": "2f9bd2e444309079cadfd27f085b0d7c3a8d01e6236d65c1e200b43409b71d81",
}


def object_info_fixture():
    combo = ["fixture-value"]

    def scalar(kind):
        return [kind, {}]

    def choice():
        return [combo, {}]

    def node(required, optional=()):
        return {
            "input": {
                "required": dict(required),
                "optional": dict(optional),
            }
        }

    return {
        "CLIPTextEncode": node(
            [("text", scalar("STRING")), ("clip", scalar("CLIP"))]
        ),
        "UNETLoader": node(
            [("unet_name", choice()), ("weight_dtype", choice())]
        ),
        "LoraLoaderModelOnly": node(
            [
                ("model", scalar("MODEL")),
                ("lora_name", choice()),
                ("strength_model", scalar("FLOAT")),
            ]
        ),
        "ModelSamplingSD3": node(
            [("model", scalar("MODEL")), ("shift", scalar("FLOAT"))]
        ),
        "KSamplerAdvanced": node(
            [
                ("model", scalar("MODEL")),
                ("add_noise", choice()),
                ("noise_seed", scalar("INT")),
                ("steps", scalar("INT")),
                ("cfg", scalar("FLOAT")),
                ("sampler_name", choice()),
                ("scheduler", choice()),
                ("positive", scalar("CONDITIONING")),
                ("negative", scalar("CONDITIONING")),
                ("latent_image", scalar("LATENT")),
                ("start_at_step", scalar("INT")),
                ("end_at_step", scalar("INT")),
                ("return_with_leftover_noise", choice()),
            ]
        ),
        "EmptyHunyuanLatentVideo": node(
            [
                ("width", scalar("INT")),
                ("height", scalar("INT")),
                ("length", scalar("INT")),
                ("batch_size", scalar("INT")),
            ]
        ),
        "WanImageToVideo": node(
            [
                ("positive", scalar("CONDITIONING")),
                ("negative", scalar("CONDITIONING")),
                ("vae", scalar("VAE")),
                ("width", scalar("INT")),
                ("height", scalar("INT")),
                ("length", scalar("INT")),
                ("batch_size", scalar("INT")),
            ],
            [
                ("clip_vision_output", scalar("CLIP_VISION_OUTPUT")),
                ("start_image", scalar("IMAGE")),
            ],
        ),
        "WanFirstLastFrameToVideo": node(
            [
                ("positive", scalar("CONDITIONING")),
                ("negative", scalar("CONDITIONING")),
                ("vae", scalar("VAE")),
                ("width", scalar("INT")),
                ("height", scalar("INT")),
                ("length", scalar("INT")),
                ("batch_size", scalar("INT")),
            ],
            [
                ("clip_vision_start_image", scalar("CLIP_VISION_OUTPUT")),
                ("clip_vision_end_image", scalar("CLIP_VISION_OUTPUT")),
                ("start_image", scalar("IMAGE")),
                ("end_image", scalar("IMAGE")),
            ],
        ),
        "CreateVideo": node(
            [("images", scalar("IMAGE")), ("fps", scalar("FLOAT"))],
            [("audio", scalar("AUDIO"))],
        ),
        "SaveVideo": node(
            [
                ("video", scalar("VIDEO")),
                ("filename_prefix", scalar("STRING")),
                ("format", choice()),
                ("codec", choice()),
            ]
        ),
        "LoadImage": node([("image", choice())]),
        "CLIPLoader": node(
            [("clip_name", choice()), ("type", choice()), ("device", choice())]
        ),
        "VAELoader": node([("vae_name", choice())]),
        "VAEDecode": node(
            [("samples", scalar("LATENT")), ("vae", scalar("VAE"))]
        ),
        "PrimitiveInt": node([("value", scalar("INT"))]),
        "PrimitiveFloat": node([("value", scalar("FLOAT"))]),
        "PrimitiveBoolean": node([("value", scalar("BOOLEAN"))]),
        "ComfySwitchNode": node(
            [
                ("on_false", scalar("*")),
                ("on_true", scalar("*")),
                ("switch", scalar("BOOLEAN")),
            ]
        ),
        "ComfyMathExpression": node(
            [("expression", scalar("STRING"))],
            [
                ("values.a", scalar("FLOAT")),
                ("values.b", scalar("FLOAT")),
                ("values.c", scalar("FLOAT")),
            ],
        ),
    }


EXPECTED_TARGETS = {
    "wan22_t2v": {
        "positive_prompt": ("CLIPTextEncode", "text"),
        "negative_prompt": ("CLIPTextEncode", "text"),
        "seed": ("KSamplerAdvanced", "noise_seed"),
        "width": ("EmptyHunyuanLatentVideo", "width"),
        "height": ("EmptyHunyuanLatentVideo", "height"),
        "frames": ("EmptyHunyuanLatentVideo", "length"),
        "output_prefix": ("SaveVideo", "filename_prefix"),
    },
    "wan22_i2v": {
        "positive_prompt": ("CLIPTextEncode", "text"),
        "negative_prompt": ("CLIPTextEncode", "text"),
        "seed": ("KSamplerAdvanced", "noise_seed"),
        "width": ("WanImageToVideo", "width"),
        "height": ("WanImageToVideo", "height"),
        "frames": ("WanImageToVideo", "length"),
        "start_image": ("LoadImage", "image"),
        "output_prefix": ("SaveVideo", "filename_prefix"),
    },
    "wan22_flf2v": {
        "positive_prompt": ("CLIPTextEncode", "text"),
        "negative_prompt": ("CLIPTextEncode", "text"),
        "seed": ("KSamplerAdvanced", "noise_seed"),
        "width": ("WanFirstLastFrameToVideo", "width"),
        "height": ("WanFirstLastFrameToVideo", "height"),
        "frames": ("WanFirstLastFrameToVideo", "length"),
        "start_image": ("LoadImage", "image"),
        "end_image": ("LoadImage", "image"),
        "output_prefix": ("SaveVideo", "filename_prefix"),
    },
}


def resolve_pointer(prompt, pointer):
    parts = pointer.strip("/").split("/")
    return prompt[parts[0]], parts[-1]


def assert_prompt_links_resolve(test_case, prompt):
    for node_id, node in prompt.items():
        for field, value in node["inputs"].items():
            if (
                isinstance(value, list)
                and len(value) == 2
                and isinstance(value[0], str)
                and isinstance(value[1], int)
            ):
                test_case.assertIn(
                    value[0],
                    prompt,
                    f"{node_id}.{field} references an omitted node {value[0]}",
                )


def effective_value(prompt, value, seen=None):
    if not (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
    ):
        return value
    seen = set() if seen is None else seen
    node_id = value[0]
    if node_id in seen:
        raise AssertionError(f"cycle while resolving {node_id}")
    seen.add(node_id)
    node = prompt[node_id]
    if node["class_type"] in {"PrimitiveInt", "PrimitiveFloat", "PrimitiveBoolean"}:
        return effective_value(prompt, node["inputs"]["value"], seen)
    if node["class_type"] == "ComfySwitchNode":
        enabled = effective_value(prompt, node["inputs"]["switch"], seen.copy())
        selected = "on_true" if enabled else "on_false"
        return effective_value(prompt, node["inputs"][selected], seen)
    return value


def converted_workflow(mode):
    workflow = json.loads(
        (FIXTURES / WORKFLOWS[mode]).read_text(encoding="utf-8")
    )
    prompt = prepare_workflows.convert(workflow, object_info_fixture())
    return prompt, prepare_workflows.build_bindings(prompt, mode)


def converted_workflow_with_turbo(mode):
    workflow = json.loads(
        (FIXTURES / WORKFLOWS[mode]).read_text(encoding="utf-8")
    )
    prompt, turbo = prepare_workflows.convert(
        workflow, object_info_fixture(), include_metadata=True
    )
    bindings = prepare_workflows.build_bindings(prompt, mode)
    if turbo is not None:
        bindings["_turbo"] = turbo
    return prompt, bindings


class PinnedWorkflowConversionTests(unittest.TestCase):
    def test_fixtures_match_pinned_commit_and_dockerfile(self):
        dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
        fixture_readme = (FIXTURES / "README.md").read_text(encoding="utf-8")
        self.assertIn(f"ARG WORKFLOW_TEMPLATES_COMMIT={PINNED_COMMIT}", dockerfile)
        self.assertIn(PINNED_COMMIT, fixture_readme)
        for filename, expected_hash in PINNED_CANONICAL_SHA256.items():
            self.assertIn(PINNED_SHA256[filename], fixture_readme)
            parsed = json.loads((FIXTURES / filename).read_text(encoding="utf-8"))
            canonical = json.dumps(
                parsed, ensure_ascii=True, separators=(",", ":"), sort_keys=True
            ).encode()
            self.assertEqual(hashlib.sha256(canonical).hexdigest(), expected_hash, filename)

    def test_each_pinned_workflow_converts_to_correct_bindings(self):
        info = object_info_fixture()
        for mode, filename in WORKFLOWS.items():
            with self.subTest(mode=mode):
                workflow = json.loads((FIXTURES / filename).read_text(encoding="utf-8"))
                prompt = prepare_workflows.convert(workflow, info)
                bindings = prepare_workflows.build_bindings(prompt, mode)
                assert_prompt_links_resolve(self, prompt)
                for name in (
                    "positive_prompt",
                    "negative_prompt",
                    "seed",
                    "width",
                    "height",
                    "frames",
                    "fps",
                    "steps",
                    "cfg",
                    "output_prefix",
                    "high_model_target",
                    "low_model_target",
                ):
                    self.assertIn(name, bindings)
                for binding, (expected_class, expected_field) in EXPECTED_TARGETS[mode].items():
                    node, field = resolve_pointer(prompt, bindings[binding])
                    self.assertEqual(node["class_type"], expected_class, binding)
                    self.assertEqual(field, expected_field, binding)
                    self.assertIn(field, node["inputs"], binding)
                if mode in {"wan22_t2v", "wan22_i2v"}:
                    self.assertTrue(
                        any(node_id.startswith("subgraph:") for node_id in prompt)
                    )
                else:
                    self.assertFalse(
                        any(node_id.startswith("subgraph:") for node_id in prompt)
                    )
                    self.assertNotIn("61", prompt, "disabled Lightning output leaked")

                save_nodes = [
                    node for node in prompt.values() if node["class_type"] == "SaveVideo"
                ]
                self.assertEqual(len(save_nodes), 1)
                self.assertEqual(save_nodes[0]["inputs"]["format"], "mp4")
                self.assertEqual(save_nodes[0]["inputs"]["codec"], "auto")
                self.assertIn("filename_prefix", save_nodes[0]["inputs"])
                video_source = save_nodes[0]["inputs"]["video"][0]
                self.assertEqual(prompt[video_source]["class_type"], "CreateVideo")

    def test_optional_official_lightning_loras_are_bypassed_semantically(self):
        for mode in ("wan22_t2v", "wan22_i2v"):
            with self.subTest(mode=mode):
                prompt, bindings = converted_workflow(mode)
                lightning = [
                    node
                    for node in prompt.values()
                    if node["class_type"] == "LoraLoaderModelOnly"
                    and "lightx2v" in str(node["inputs"].get("lora_name", "")).lower()
                ]
                self.assertEqual(lightning, [])
                self.assertNotIn("lightx2v", json.dumps(prompt).lower())
                model_switches = []
                for node_id, node in prompt.items():
                    if node["class_type"] != "ComfySwitchNode":
                        continue
                    if any(
                        consumer["class_type"] == "ModelSamplingSD3"
                        and consumer["inputs"].get("model") == [node_id, 0]
                        for consumer in prompt.values()
                    ):
                        model_switches.append(node)
                self.assertEqual(model_switches, [])
                for branch in ("high", "low"):
                    target = bindings[f"{branch}_model_target"].split("/")[1]
                    self.assertEqual(prompt[target]["class_type"], "ModelSamplingSD3")
                    raw = prompt[prompt[target]["inputs"]["model"][0]]
                    self.assertEqual(raw["class_type"], "UNETLoader")
                    self.assertIn(
                        f"{branch}_noise", raw["inputs"]["unet_name"].lower()
                    )

    def test_strict_binding_failure_is_preserved(self):
        workflow = json.loads(
            (FIXTURES / WORKFLOWS["wan22_t2v"]).read_text(encoding="utf-8")
        )
        prompt = prepare_workflows.convert(workflow, object_info_fixture())
        for node in prompt.values():
            if node["class_type"] == "SaveVideo":
                node["inputs"].pop("filename_prefix", None)
        with self.assertRaisesRegex(RuntimeError, "output_prefix"):
            prepare_workflows.build_bindings(prompt, "wan22_t2v")

    def test_non_default_steps_and_cfg_control_both_sampling_phases(self):
        for mode in WORKFLOWS:
            with self.subTest(mode=mode):
                prompt, bindings = converted_workflow(mode)
                with mock.patch(
                    "app.comfy.load_template", return_value=(prompt, bindings)
                ):
                    bound, _ = bind_workflow(
                        mode, {"steps": 33, "cfg": 7.5}
                    )
                samplers = prepare_workflows.ordered_samplers(bound)
                self.assertEqual(len(samplers), 2)
                first, second = samplers[0][1], samplers[1][1]
                for sampler in (first, second):
                    self.assertEqual(
                        effective_value(bound, sampler["inputs"]["steps"]), 33
                    )
                    self.assertEqual(
                        effective_value(bound, sampler["inputs"]["cfg"]), 7.5
                    )
                self.assertEqual(
                    effective_value(bound, first["inputs"]["end_at_step"]), 16
                )
                self.assertEqual(
                    effective_value(bound, second["inputs"]["start_at_step"]), 16
                )
                self.assertEqual(
                    effective_value(bound, second["inputs"]["end_at_step"]), 33
                )

    def test_user_loras_are_injected_on_each_sampler_model_path(self):
        for mode in WORKFLOWS:
            with self.subTest(mode=mode):
                prompt, bindings = converted_workflow(mode)
                self.assertEqual(
                    prompt[bindings["high_model_target"].split("/")[1]]["class_type"],
                    "ModelSamplingSD3",
                )
                self.assertEqual(
                    prompt[bindings["low_model_target"].split("/")[1]]["class_type"],
                    "ModelSamplingSD3",
                )
                inject_loras(
                    prompt,
                    bindings,
                    [{"filename": "fake-high.safetensors", "weight": 0.75}],
                    "high",
                )
                inject_loras(
                    prompt,
                    bindings,
                    [{"filename": "fake-low.safetensors", "weight": 0.5}],
                    "low",
                )
                samplers = prepare_workflows.ordered_samplers(prompt)
                expected = (
                    (samplers[0][1], "fake-high.safetensors"),
                    (samplers[1][1], "fake-low.safetensors"),
                )
                for sampler, filename in expected:
                    sampling_node = prompt[sampler["inputs"]["model"][0]]
                    self.assertEqual(sampling_node["class_type"], "ModelSamplingSD3")
                    injected = prompt[sampling_node["inputs"]["model"][0]]
                    self.assertEqual(injected["class_type"], "LoraLoaderModelOnly")
                    self.assertEqual(injected["inputs"]["lora_name"], filename)
                    upstream = prompt[injected["inputs"]["model"][0]]
                    self.assertEqual(upstream["class_type"], "UNETLoader")

    def test_turbo_recipe_and_runtime_graph_match_pinned_t2v_i2v_topology(self):
        expected_names = {
            "wan22_t2v": ("wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors", "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors"),
            "wan22_i2v": ("wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"),
        }
        for mode, names in expected_names.items():
            with self.subTest(mode=mode):
                prompt, bindings = converted_workflow_with_turbo(mode)
                recipe = bindings["_turbo"]
                self.assertEqual(recipe["sampling"], {"steps": 4, "split_step": 2, "cfg": 1})
                self.assertEqual(
                    (recipe["branches"]["high"]["lora_name"], recipe["branches"]["low"]["lora_name"]),
                    names,
                )
                with (
                    mock.patch("app.comfy.load_template", return_value=(prompt, bindings)),
                    mock.patch("app.comfy.check_turbo_assets", return_value=(True, {})),
                ):
                    bound, runtime_bindings = bind_workflow(
                        mode, {"steps": 4, "cfg": 1.0, "turbo_mode": True}
                    )
                self.assertFalse(any(node["class_type"] == "ComfySwitchNode" for node in bound.values()))
                self.assertEqual(
                    sum(
                        node["class_type"] == "LoraLoaderModelOnly"
                        and "lightx2v" in str(node["inputs"].get("lora_name", "")).lower()
                        for node in bound.values()
                    ),
                    2,
                )
                inject_loras(bound, runtime_bindings, [{"filename": "user-high.safetensors", "weight": 0.8}], "high")
                inject_loras(bound, runtime_bindings, [{"filename": "user-low.safetensors", "weight": 0.7}], "low")
                for branch, user_name, lightx_name in (
                    ("high", "user-high.safetensors", names[0]),
                    ("low", "user-low.safetensors", names[1]),
                ):
                    sampling = bound[runtime_bindings[f"{branch}_model_target"].split("/")[1]]
                    user_lora = bound[sampling["inputs"]["model"][0]]
                    lightx = bound[user_lora["inputs"]["model"][0]]
                    raw = bound[lightx["inputs"]["model"][0]]
                    self.assertEqual(user_lora["inputs"]["lora_name"], user_name)
                    self.assertEqual(lightx["inputs"]["lora_name"], lightx_name)
                    self.assertEqual(raw["class_type"], "UNETLoader")
                samplers = prepare_workflows.ordered_samplers(bound)
                for _, sampler in samplers:
                    self.assertEqual(effective_value(bound, sampler["inputs"]["steps"]), 4)
                    self.assertEqual(effective_value(bound, sampler["inputs"]["cfg"]), 1)
                self.assertEqual(effective_value(bound, samplers[0][1]["inputs"]["end_at_step"]), 2)
                self.assertEqual(effective_value(bound, samplers[1][1]["inputs"]["start_at_step"]), 2)

    def test_missing_turbo_assets_fail_before_prompt_materialization(self):
        prompt, bindings = converted_workflow_with_turbo("wan22_t2v")
        with (
            mock.patch("app.comfy.load_template", return_value=(prompt, bindings)),
            mock.patch("app.comfy.check_turbo_assets") as checked,
        ):
            standard, _ = bind_workflow("wan22_t2v", {"turbo_mode": False})
        checked.assert_not_called()
        self.assertNotIn("lightx2v", json.dumps(standard).lower())
        with (
            mock.patch("app.comfy.load_template", return_value=(prompt, bindings)),
            mock.patch("app.comfy.check_turbo_assets", return_value=(False, {"wan22_t2v_lightx_high": "missing"})),
        ):
            with self.assertRaisesRegex(RuntimeError, "install profile t2v-turbo"):
                bind_workflow("wan22_t2v", {"turbo_mode": True})

    def test_reusable_requires_current_converter_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "workflow.api.json"
            source_hash = "fixture-source-hash"
            base = {"_bindings": {"seed": "/1/inputs/seed"}, "_source_sha256": source_hash}
            destination.write_text(json.dumps(base), encoding="utf-8")
            self.assertFalse(prepare_workflows.reusable(destination, source_hash))
            base["_converter_schema_version"] = prepare_workflows.CONVERTER_SCHEMA_VERSION
            destination.write_text(json.dumps(base), encoding="utf-8")
            self.assertTrue(prepare_workflows.reusable(destination, source_hash))
            base["_converter_schema_version"] -= 1
            destination.write_text(json.dumps(base), encoding="utf-8")
            self.assertFalse(prepare_workflows.reusable(destination, source_hash))

    def test_converter_metadata_is_not_sent_as_a_comfy_node(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = {
                "1": {"class_type": "FixtureNode", "inputs": {}},
                "_bindings": {"seed": "/1/inputs/seed"},
                "_source_sha256": "source",
                "_converter_schema_version": prepare_workflows.CONVERTER_SCHEMA_VERSION,
                "_turbo": {"branches": {}, "sampling": {}},
            }
            (root / "fixture.api.json").write_text(
                json.dumps(document), encoding="utf-8"
            )
            with mock.patch(
                "app.comfy.get_settings",
                return_value=SimpleNamespace(workflow_dir=root),
            ):
                prompt, bindings = load_template("fixture")
            self.assertEqual(set(prompt), {"1"})
            self.assertEqual(bindings["seed"], document["_bindings"]["seed"])
            self.assertEqual(bindings["_turbo"], document["_turbo"])

    def test_prepare_main_writes_valid_api_json_with_bindings(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            with (
                mock.patch.object(prepare_workflows, "SOURCE", FIXTURES),
                mock.patch.object(prepare_workflows, "DEST", destination),
                mock.patch.object(
                    prepare_workflows,
                    "get_json",
                    return_value=object_info_fixture(),
                ),
            ):
                prepare_workflows.main([])
            for mode in WORKFLOWS:
                generated = destination / f"{mode}.api.json"
                parsed = json.loads(generated.read_text(encoding="utf-8"))
                self.assertTrue(parsed["_bindings"])
                if mode in {"wan22_t2v", "wan22_i2v"}:
                    self.assertEqual(
                        parsed["_turbo"]["sampling"],
                        {"steps": 4, "split_step": 2, "cfg": 1},
                    )
                else:
                    self.assertNotIn("_turbo", parsed)
                self.assertEqual(
                    parsed["_converter_schema_version"],
                    prepare_workflows.CONVERTER_SCHEMA_VERSION,
                )
                prompt = {
                    key: value for key, value in parsed.items() if not key.startswith("_")
                }
                assert_prompt_links_resolve(self, prompt)
                self.assertEqual(
                    parsed["_source_sha256"],
                    hashlib.sha256((FIXTURES / WORKFLOWS[mode]).read_bytes()).hexdigest(),
                )


class EntrypointStaticSmokeTests(unittest.TestCase):
    def test_entrypoint_is_path_independent_and_network_bindings_are_scoped(self):
        entrypoint = (PROJECT_ROOT / "scripts/entrypoint.sh").read_text(encoding="utf-8")
        self.assertIn('export PATH="$VENV_BIN:${PATH:-', entrypoint)
        self.assertIn('PYTHON_BIN="$VENV_BIN/python"', entrypoint)
        self.assertIn('UVICORN_BIN="$VENV_BIN/uvicorn"', entrypoint)
        self.assertIn('"$PYTHON_BIN" /opt/ComfyUI/main.py', entrypoint)
        self.assertIn('cd "$APP_ROOT"', entrypoint)
        self.assertIn('"$PYTHON_BIN" "$APP_ROOT/scripts/prepare_workflows.py"', entrypoint)
        self.assertIn('"$UVICORN_BIN" --app-dir "$APP_ROOT" app.main:app --host 0.0.0.0', entrypoint)
        self.assertIn('COMFYUI_HOST="${COMFYUI_HOST:-127.0.0.1}"', entrypoint)

    def test_only_explicit_comfy_runtime_directories_become_writable(self):
        dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("/opt/ComfyUI/user /opt/ComfyUI/temp", dockerfile)
        self.assertIn(
            "chown appuser:appuser /opt/ComfyUI/user /opt/ComfyUI/temp",
            dockerfile,
        )
        self.assertNotIn("chown -R appuser:appuser /opt/ComfyUI", dockerfile)
        self.assertNotIn("chown -R appuser:appuser /workspace", dockerfile)

    def test_entrypoint_initializes_every_required_workspace_parent(self):
        entrypoint = (PROJECT_ROOT / "scripts/entrypoint.sh").read_text(
            encoding="utf-8"
        )
        for path in (
            '"$DATA_ROOT/models"',
            '"$DATA_ROOT/inputs"',
            '"$DATA_ROOT/outputs"',
            '"$DATA_ROOT/jobs"',
            '"$DATA_ROOT/cache"',
            '"$DATA_ROOT/downloads"',
            '"$WORKFLOW_DIR"',
        ):
            self.assertIn(path, entrypoint)
        self.assertIn("install -d -o appuser -g appuser -m 775", entrypoint)
        self.assertNotIn("chown -R", entrypoint)

    def test_runtime_smoke_covers_container_only_checks(self):
        smoke = (PROJECT_ROOT / "scripts/smoke_runtime.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("env -i PATH=/usr/bin:/bin", smoke)
        self.assertIn("gosu appuser test -w", smoke)
        self.assertIn("wan22_flf2v", smoke)
        self.assertIn("incompatible SaveVideo format/codec", smoke)
        self.assertIn("optional Lightning LoRA remains", smoke)
        self.assertIn("/health/live", smoke)
        self.assertIn("/health/ready", smoke)

    def test_health_routes_remain_distinct(self):
        main = (PROJECT_ROOT / "app/main.py").read_text(encoding="utf-8")
        self.assertIn('@app.get("/health/live")', main)
        self.assertIn('@app.get("/health/ready")', main)
        for field in ("api", "database", "directories", "comfyui", "workflows", "models", "profile"):
            self.assertIn(f'"{field}"', main)


if __name__ == "__main__":
    unittest.main()
