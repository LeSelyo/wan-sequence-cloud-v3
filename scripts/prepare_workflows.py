"""Convert official ComfyUI UI workflows to API prompts and add semantic bindings.

The conversion asks the running ComfyUI instance for its node definitions, so the
serialized widget order stays aligned with the installed server version.
"""
from __future__ import annotations

import json
import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Any


COMFY_URL = os.getenv(
    "COMFY_URL",
    f"http://{os.getenv('COMFYUI_HOST', '127.0.0.1')}:{os.getenv('COMFYUI_PORT', '8188')}",
).rstrip("/")
SOURCE = Path(os.getenv("UI_WORKFLOW_DIR", "/app/ui_workflows"))
DATA_ROOT = Path(os.getenv("DATA_ROOT", "/workspace"))
DEST = Path(os.getenv("WORKFLOW_DIR", str(DATA_ROOT / "cache/workflows")))
FILES = {
    "wan22_t2v": "video_wan2_2_14B_t2v.json",
    "wan22_i2v": "video_wan2_2_14B_i2v.json",
    "wan22_flf2v": "video_wan2_2_14B_flf2v.json",
    "wan22_animate": "video_wan2_2_14B_animate.json",
}
# wan22_animate is structurally unlike the other three: it has DWPose/SAM2
# preprocessing nodes, a PointsEditor, and three chained subgraph instances
# (one sampler + two "Video Extend" copies). expand_subgraphs() and
# build_bindings() now handle this shape (subgraph-to-subgraph wiring,
# WanAnimateToVideo's width/height/reference_image, LoadVideo's driving
# file) and it converts and binds cleanly against a live object_info with the
# custom node packs installed (verified 2026-09-28). Still listed in
# EXPERIMENTAL: total clip length is architecturally fixed by the template's
# 3-stage chain (shot.frames is silently ignored, see build_bindings' comment
# below) and ANIMATE_MIX/ANIMATE_MOVE differ only by the background_inputs binding
# (see build_bindings). Kept here so a regression in either area is caught rather than
# blocking startup for t2v/i2v/flf2v/image — see main().
EXPERIMENTAL = {"wan22_animate"}
CONVERTER_SCHEMA_VERSION = 7


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reusable(destination: Path, source_hash: str) -> bool:
    if not destination.is_file() or destination.stat().st_size == 0:
        return False
    try:
        existing = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        bool(existing.get("_bindings"))
        and existing.get("_source_sha256") == source_hash
        and existing.get("_converter_schema_version") == CONVERTER_SCHEMA_VERSION
    )


def primitive(spec: Any) -> bool:
    if isinstance(spec, list):
        return True
    return spec in {"STRING", "INT", "FLOAT", "BOOLEAN"}


def normalized_link(link: Any) -> dict[str, Any]:
    if isinstance(link, dict):
        return {
            "id": link["id"],
            "origin_id": link["origin_id"],
            "origin_slot": link["origin_slot"],
            "target_id": link["target_id"],
            "target_slot": link["target_slot"],
            "type": link.get("type"),
        }
    return {
        "id": link[0],
        "origin_id": link[1],
        "origin_slot": link[2],
        "target_id": link[3],
        "target_slot": link[4],
        "type": link[5] if len(link) > 5 else None,
    }


def _copy_node(node: dict, node_prefix: str, link_prefix: str) -> dict:
    copied = deepcopy(node)
    copied["id"] = f"{node_prefix}{node['id']}"
    for item in copied.get("inputs", []):
        if item.get("link") is not None:
            item["link"] = f"{link_prefix}{item['link']}"
    return copied


def is_node_reference(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
    )


def _replace_reference(prompt: dict, old: list[Any], new: Any) -> None:
    for node in prompt.values():
        for field, value in node["inputs"].items():
            if value == old:
                node["inputs"][field] = deepcopy(new)


def _prune_unreachable(prompt: dict) -> None:
    """Keep only dependencies reachable from ComfyUI output nodes."""
    roots = [
        node_id
        for node_id, node in prompt.items()
        if node["class_type"] in {"SaveVideo", "VHS_VideoCombine", "SaveImage"}
    ]
    reachable: set[str] = set()
    pending = list(roots)
    while pending:
        node_id = pending.pop()
        if node_id in reachable or node_id not in prompt:
            continue
        reachable.add(node_id)
        for value in prompt[node_id]["inputs"].values():
            referenced = referenced_node(value)
            if referenced is not None:
                pending.append(referenced)
    for node_id in set(prompt) - reachable:
        prompt.pop(node_id)


def prepare_runtime_compatibility(prompt: dict) -> dict[str, Any] | None:
    """Extract the official turbo recipe, then materialize a pruned standard graph."""
    model_switches: list[tuple[str, dict, dict, str]] = []
    selector = None
    for node_id, node in list(prompt.items()):
        if node["class_type"] == "SaveVideo":
            node["inputs"]["format"] = "mp4"
            node["inputs"]["codec"] = "auto"
            continue
        if node["class_type"] != "ComfySwitchNode":
            continue
        inputs = node["inputs"]
        raw_reference = inputs.get("on_false")
        lightning_reference = inputs.get("on_true")
        if not is_node_reference(raw_reference) or not is_node_reference(
            lightning_reference
        ):
            continue
        raw_node = prompt.get(raw_reference[0])
        lightning_node = prompt.get(lightning_reference[0])
        if (
            not raw_node
            or raw_node["class_type"] != "UNETLoader"
            or not lightning_node
            or lightning_node["class_type"] != "LoraLoaderModelOnly"
            or lightning_node["inputs"].get("model") != raw_reference
            or "lightx2v"
            not in str(lightning_node["inputs"].get("lora_name", "")).lower()
        ):
            continue
        switch_output = [node_id, 0]
        feeds_model_sampling = any(
            consumer["class_type"] == "ModelSamplingSD3"
            and consumer["inputs"].get("model") == switch_output
            for consumer in prompt.values()
        )
        if not feeds_model_sampling:
            continue
        sampling_ids = [
            consumer_id
            for consumer_id, consumer in prompt.items()
            if consumer["class_type"] == "ModelSamplingSD3"
            and consumer["inputs"].get("model") == switch_output
        ]
        if len(sampling_ids) != 1:
            continue
        current_selector = inputs.get("switch")
        if selector is not None and current_selector != selector:
            raise RuntimeError("LightX2V model switches do not share one selector")
        selector = current_selector
        model_switches.append((node_id, raw_node, lightning_node, sampling_ids[0]))

    if not model_switches:
        return None
    if len(model_switches) != 2 or not is_node_reference(selector):
        raise RuntimeError("expected two LightX2V branches with a shared selector")

    branches: dict[str, dict[str, Any]] = {}
    for switch_id, raw_node, lightning_node, sampling_id in model_switches:
        raw_reference = prompt[switch_id]["inputs"]["on_false"]
        filename = str(raw_node["inputs"].get("unet_name", "")).lower()
        branch = "high" if "high_noise" in filename else "low" if "low_noise" in filename else ""
        if not branch or branch in branches:
            raise RuntimeError("cannot identify unique high/low LightX2V branches")
        branches[branch] = {
            "model_target": pointer(sampling_id, "model"),
            "lora_name": lightning_node["inputs"]["lora_name"],
            "strength_model": lightning_node["inputs"].get("strength_model", 1.0),
        }

    sampling: dict[str, Any] = {}
    for switch_id, node in list(prompt.items()):
        if node["class_type"] != "ComfySwitchNode" or node["inputs"].get("switch") != selector:
            continue
        consumers = [
            field
            for consumer in prompt.values()
            if consumer["class_type"] in {"KSampler", "KSamplerAdvanced"}
            for field, value in consumer["inputs"].items()
            if value == [switch_id, 0]
        ]
        turbo_value = resolved_value(prompt, node["inputs"].get("on_true"))
        if "start_at_step" in consumers:
            sampling["split_step"] = turbo_value
        elif "steps" in consumers:
            sampling["steps"] = turbo_value
        if "cfg" in consumers:
            sampling["cfg"] = turbo_value
        _replace_reference(prompt, [switch_id, 0], node["inputs"]["on_false"])

    if set(branches) != {"high", "low"} or set(sampling) != {"steps", "cfg", "split_step"}:
        raise RuntimeError("incomplete LightX2V branch or sampling recipe")
    _prune_unreachable(prompt)
    return {"branches": branches, "sampling": sampling}


def expand_subgraphs(workflow: dict) -> dict:
    """Flatten one-level frontend subgraphs into a normal ComfyUI graph.

    Subgraph input/output pseudo-nodes (-10/-20) are frontend-only. External
    links are rewired directly to their internal targets and outputs. Inputs
    exposed only as proxy widgets fall back to the serialized internal widget.
    """
    raw_subgraphs = workflow.get("definitions", {}).get("subgraphs", [])
    if isinstance(raw_subgraphs, dict):
        subgraphs = {str(key): value for key, value in raw_subgraphs.items()}
        subgraphs.update(
            {str(value.get("id")): value for value in raw_subgraphs.values()}
        )
    else:
        subgraphs = {str(item["id"]): item for item in raw_subgraphs}
    if not subgraphs:
        return workflow

    main_nodes = {str(node["id"]): node for node in workflow.get("nodes", [])}
    instances = {
        node_id: node
        for node_id, node in main_nodes.items()
        if str(node.get("type")) in subgraphs
    }
    if not instances:
        return workflow

    main_links = [normalized_link(link) for link in workflow.get("links", [])]
    main_links_by_id = {str(link["id"]): link for link in main_links}
    flat_nodes: list[dict] = []
    flat_links: list[dict] = []

    def resolve_subgraph_output(
        instance_id: str, slot: int, _seen: frozenset[str] = frozenset()
    ) -> tuple[str, int]:
        """Trace a subgraph instance's exposed output back to the real (flat)
        node+slot that produces it, recursing through chained subgraph
        instances (one subgraph instance's output feeding another instance's
        input directly, as in Wan 2.2 Animate's Sampling -> Extend -> Extend
        chain)."""
        if instance_id in _seen:
            raise RuntimeError(f"cyclic subgraph output passthrough at {instance_id}")
        instance = instances[instance_id]
        subgraph = subgraphs[str(instance["type"])]
        local_links = [normalized_link(link) for link in subgraph.get("links", [])]
        match = next(
            link
            for link in local_links
            if str(link["target_id"]) == "-20" and int(link["target_slot"]) == slot
        )
        origin_id = str(match["origin_id"])
        origin_slot = int(match["origin_slot"])
        if origin_id != "-10":
            return f"subgraph:{instance_id}:{origin_id}", origin_slot
        subgraph_input = subgraph.get("inputs", [])[origin_slot]
        instance_input = next(
            (
                item
                for item in instance.get("inputs", [])
                if item.get("name") == subgraph_input.get("name")
            ),
            None,
        )
        external_link_id = instance_input.get("link") if instance_input else None
        if external_link_id is None:
            raise RuntimeError("subgraph output passes through an unconnected input")
        external = main_links_by_id[str(external_link_id)]
        if str(external["origin_id"]) in instances:
            return resolve_subgraph_output(
                str(external["origin_id"]),
                int(external["origin_slot"]),
                _seen | {instance_id},
            )
        return str(external["origin_id"]), int(external["origin_slot"])

    for node_id, node in main_nodes.items():
        if node_id not in instances:
            flat_nodes.append(_copy_node(node, "", "main:"))

    for link in main_links:
        if (
            str(link["origin_id"]) not in instances
            and str(link["target_id"]) not in instances
        ):
            flat_links.append(
                {
                    **link,
                    "id": f"main:{link['id']}",
                    "origin_id": str(link["origin_id"]),
                    "target_id": str(link["target_id"]),
                }
            )

    for instance_id, instance in instances.items():
        subgraph = subgraphs[str(instance["type"])]
        nested_types = {
            str(node.get("type")) for node in subgraph.get("nodes", [])
        } & set(subgraphs)
        if nested_types:
            raise RuntimeError(
                f"nested subgraphs are not supported: {sorted(nested_types)}"
            )
        node_prefix = f"subgraph:{instance_id}:"
        link_prefix = f"subgraph:{instance_id}:link:"
        copied_nodes = {
            str(node["id"]): _copy_node(node, node_prefix, link_prefix)
            for node in subgraph.get("nodes", [])
        }
        flat_nodes.extend(copied_nodes.values())
        subgraph_links = [
            normalized_link(link) for link in subgraph.get("links", [])
        ]

        for link in subgraph_links:
            origin_id = str(link["origin_id"])
            target_id = str(link["target_id"])
            if origin_id not in {"-10"} and target_id not in {"-20"}:
                flat_links.append(
                    {
                        **link,
                        "id": f"{link_prefix}{link['id']}",
                        "origin_id": f"{node_prefix}{origin_id}",
                        "target_id": f"{node_prefix}{target_id}",
                    }
                )
                continue

            if origin_id == "-10":
                input_slot = int(link["origin_slot"])
                subgraph_input = subgraph.get("inputs", [])[input_slot]
                instance_input = next(
                    (
                        item
                        for item in instance.get("inputs", [])
                        if item.get("name") == subgraph_input.get("name")
                    ),
                    None,
                )
                external_link_id = (
                    instance_input.get("link") if instance_input else None
                )
                target_node = copied_nodes[target_id]
                target_node.get("inputs", [])[int(link["target_slot"])]["link"] = (
                    f"{link_prefix}{link['id']}"
                    if external_link_id is not None
                    else None
                )
                if external_link_id is not None:
                    external = main_links_by_id[str(external_link_id)]
                    if str(external["origin_id"]) in instances:
                        real_origin_id, real_origin_slot = resolve_subgraph_output(
                            str(external["origin_id"]), int(external["origin_slot"])
                        )
                    else:
                        real_origin_id = str(external["origin_id"])
                        real_origin_slot = external["origin_slot"]
                    flat_links.append(
                        {
                            **link,
                            "id": f"{link_prefix}{link['id']}",
                            "origin_id": real_origin_id,
                            "origin_slot": real_origin_slot,
                            "target_id": f"{node_prefix}{target_id}",
                        }
                    )
                continue

            output_slot = int(link["target_slot"])
            for external in main_links:
                if (
                    str(external["origin_id"]) == instance_id
                    and int(external["origin_slot"]) == output_slot
                ):
                    if str(external["target_id"]) in instances:
                        # Handled by that other instance's own input-side
                        # resolution (resolve_subgraph_output), which traces
                        # back to this same internal node -- adding it here
                        # too would create a duplicate link.
                        continue
                    flat_links.append(
                        {
                            **external,
                            "id": f"main:{external['id']}",
                            "origin_id": f"{node_prefix}{origin_id}",
                            "origin_slot": link["origin_slot"],
                            "target_id": str(external["target_id"]),
                        }
                    )

    return {**workflow, "nodes": flat_nodes, "links": flat_links, "definitions": {}}


def convert(
    workflow: dict, object_info: dict, *, include_metadata: bool = False
) -> dict | tuple[dict, dict[str, Any] | None]:
    workflow = expand_subgraphs(workflow)
    nodes = {str(node["id"]): node for node in workflow["nodes"]}
    links = {
        str(normalized_link(link)["id"]): normalized_link(link)
        for link in workflow.get("links", [])
    }

    def origin(link_id: Any) -> list[Any]:
        link = links[str(link_id)]
        node_id, slot = str(link["origin_id"]), int(link["origin_slot"])
        node = nodes[node_id]
        if node.get("type") in {"Reroute", "PrimitiveNode"} and node.get("inputs"):
            upstream = node["inputs"][0].get("link")
            if upstream is not None:
                return origin(upstream)
        return [node_id, slot]

    prompt: dict[str, dict] = {}
    for node_id, node in nodes.items():
        class_type = node.get("type")
        if node.get("mode", 0) == 4 or class_type not in object_info:
            continue
        definition = object_info[class_type].get("input", {})
        ordered = list(definition.get("required", {}).items()) + list(definition.get("optional", {}).items())
        linked = {item["name"]: item.get("link") for item in node.get("inputs", [])}
        widgets = list(node.get("widgets_values") or [])
        widget_index = 0
        inputs: dict[str, Any] = {}
        for name, spec in ordered:
            spec_type = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
            widget_value = None
            has_widget = primitive(spec_type)
            if has_widget:
                while widget_index < len(widgets) and isinstance(
                    widgets[widget_index], dict
                ):
                    widget_index += 1
                if widget_index < len(widgets):
                    widget_value = widgets[widget_index]
                    widget_index += 1
                    if name in {"seed", "noise_seed"}:
                        while widget_index < len(widgets) and (
                            isinstance(widgets[widget_index], dict)
                            or widgets[widget_index]
                            in {"fixed", "increment", "decrement", "randomize"}
                        ):
                            widget_index += 1
            if linked.get(name) is not None:
                inputs[name] = origin(linked[name])
            elif has_widget and widget_value is not None:
                # Seed widgets can serialize a control-after-generate object; only the scalar is an input.
                if not isinstance(widget_value, dict):
                    inputs[name] = widget_value
        prompt[node_id] = {"class_type": class_type, "inputs": inputs}
        if node.get("title"):
            prompt[node_id]["_meta"] = {"title": node["title"]}
    turbo = prepare_runtime_compatibility(prompt)
    return (prompt, turbo) if include_metadata else prompt


def pointer(node_id: str, field: str) -> str:
    return f"/{node_id}/inputs/{field}"


def referenced_node(value: Any) -> str | None:
    if is_node_reference(value):
        return value[0]
    return None


def resolved_value(prompt: dict, value: Any, seen: set[str] | None = None) -> Any:
    """Resolve static primitives and the selected side of a Comfy switch."""
    node_id = referenced_node(value)
    if node_id is None:
        return value
    seen = set() if seen is None else seen
    if node_id in seen or node_id not in prompt:
        return None
    seen.add(node_id)
    node = prompt[node_id]
    inputs = node["inputs"]
    if node["class_type"] in {"PrimitiveInt", "PrimitiveFloat", "PrimitiveBoolean"}:
        return resolved_value(prompt, inputs.get("value"), seen)
    if node["class_type"] == "ComfySwitchNode":
        selected = "on_true" if resolved_value(prompt, inputs.get("switch"), seen) else "on_false"
        return resolved_value(prompt, inputs.get(selected), seen)
    return None


def writable_value_pointer(
    prompt: dict, value: Any, seen: set[str] | None = None
) -> str | None:
    """Find the selected primitive behind a static Comfy switch."""
    node_id = referenced_node(value)
    if node_id is None:
        return None
    seen = set() if seen is None else seen
    if node_id in seen or node_id not in prompt:
        return None
    seen.add(node_id)
    node = prompt[node_id]
    inputs = node["inputs"]
    if node["class_type"] in {"PrimitiveInt", "PrimitiveFloat", "PrimitiveBoolean"}:
        return pointer(node_id, "value")
    if node["class_type"] == "ComfySwitchNode":
        selected = "on_true" if resolved_value(prompt, inputs.get("switch")) else "on_false"
        return writable_value_pointer(prompt, inputs.get(selected), seen)
    return None


def ordered_samplers(prompt: dict) -> list[tuple[str, dict]]:
    samplers = [
        (node_id, node)
        for node_id, node in prompt.items()
        if node["class_type"] in {"KSampler", "KSamplerAdvanced"}
    ]
    return sorted(
        samplers,
        key=lambda item: (
            item[1]["inputs"].get("add_noise") != "enable",
            item[0],
        ),
    )


def sampling_bindings(prompt: dict) -> tuple[dict, dict, str | None]:
    """Bind both phases and preserve their split when total steps changes."""
    samplers = ordered_samplers(prompt)
    if not samplers:
        return {}, {}, None
    seed_pointer = None
    for node_id, node in samplers:
        for field in ("noise_seed", "seed"):
            if field in node["inputs"]:
                seed_pointer = pointer(node_id, field)
                break
        if seed_pointer:
            break

    step_targets: list[str] = []
    shared_steps = {json.dumps(node["inputs"].get("steps")) for _, node in samplers}
    if len(shared_steps) == 1:
        upstream = writable_value_pointer(prompt, samplers[0][1]["inputs"].get("steps"))
        if upstream:
            step_targets.append(upstream)
    if not step_targets:
        step_targets.extend(pointer(node_id, "steps") for node_id, _ in samplers)

    split_targets: list[str] = []
    split_ratio = 0.5
    if len(samplers) >= 2:
        first_id, first = samplers[0]
        second_id, second = samplers[1]
        first_end = first["inputs"].get("end_at_step")
        second_start = second["inputs"].get("start_at_step")
        if first_end == second_start:
            upstream = writable_value_pointer(prompt, first_end)
            if upstream:
                split_targets.append(upstream)
        if not split_targets:
            split_targets.extend(
                [pointer(first_id, "end_at_step"), pointer(second_id, "start_at_step")]
            )
        original_total = resolved_value(prompt, first["inputs"].get("steps"))
        original_split = resolved_value(prompt, first_end)
        if isinstance(original_total, (int, float)) and original_total > 0 and isinstance(
            original_split, (int, float)
        ):
            split_ratio = float(original_split) / float(original_total)

        second_end = second["inputs"].get("end_at_step")
        if resolved_value(prompt, second_end) != resolved_value(
            prompt, second["inputs"].get("steps")
        ):
            step_targets.append(pointer(second_id, "end_at_step"))

    cfg_targets: list[str] = []
    shared_cfg = {json.dumps(node["inputs"].get("cfg")) for _, node in samplers}
    if len(shared_cfg) == 1:
        upstream = writable_value_pointer(prompt, samplers[0][1]["inputs"].get("cfg"))
        if upstream:
            cfg_targets.append(upstream)
    if not cfg_targets:
        cfg_targets.extend(pointer(node_id, "cfg") for node_id, _ in samplers)

    return (
        {
            "targets": list(dict.fromkeys(step_targets)),
            "split_targets": list(dict.fromkeys(split_targets)),
            "split_ratio": split_ratio,
        },
        {"targets": list(dict.fromkeys(cfg_targets))},
        seed_pointer,
    )


def model_depends_on(prompt: dict, value: Any, loader_id: str, seen: set[str] | None = None) -> bool:
    node_id = referenced_node(value)
    if node_id is None:
        return False
    if node_id == loader_id:
        return True
    seen = set() if seen is None else seen
    if node_id in seen or node_id not in prompt:
        return False
    seen.add(node_id)
    return any(
        model_depends_on(prompt, candidate, loader_id, seen.copy())
        for candidate in prompt[node_id]["inputs"].values()
    )


def validate_runtime_prompt(prompt: dict, workflow_name: str) -> None:
    save_nodes = [
        node for node in prompt.values() if node["class_type"] == "SaveVideo"
    ]
    if not save_nodes:
        raise RuntimeError(f"{workflow_name}: no SaveVideo node found")
    for node in save_nodes:
        inputs = node["inputs"]
        missing = {"video", "filename_prefix", "format", "codec"} - set(inputs)
        if missing:
            raise RuntimeError(
                f"{workflow_name}: SaveVideo missing inputs {sorted(missing)}"
            )
        if inputs["format"] != "mp4" or inputs["codec"] != "auto":
            raise RuntimeError(
                f"{workflow_name}: SaveVideo must use format=mp4 and codec=auto"
            )
    if workflow_name != "wan22_animate":
        # t2v/i2v/flf2v expose an explicit turbo/non-turbo Switch that
        # sampling_bindings()'s branch-pruning removes when turbo_mode is off;
        # a LightX2V loader surviving that means pruning failed. Wan 2.2
        # Animate's official template has no such switch -- it always bakes
        # the distilled LightX2V LoRA into the one recipe it ships (confirmed
        # on the pinned template: no turbo/non-turbo branch to detect), so its
        # presence here is expected, not a leak.
        forbidden = [
            str(node["inputs"].get("lora_name"))
            for node in prompt.values()
            if node["class_type"] == "LoraLoaderModelOnly"
            and "lightx2v" in str(node["inputs"].get("lora_name", "")).lower()
        ]
        if forbidden:
            raise RuntimeError(
                f"{workflow_name}: optional Lightning LoRA remains in runtime prompt: {forbidden}"
            )
    for node_id, node in prompt.items():
        for field, value in node["inputs"].items():
            referenced = referenced_node(value)
            if referenced is not None and referenced not in prompt:
                raise RuntimeError(
                    f"{workflow_name}: {node_id}.{field} references missing node {referenced}"
                )


def wire_sam2_negative_points(prompt: dict, object_info: dict | None = None) -> None:
    """Connects PointsEditor's red (negative) output to Sam2Segmentation.

    The official Animate template only links `positive_coords` (green, the
    person to segment); `coordinates_negative` is left unconnected (checked on
    the template JSON 2026-10-02), so a negative point -- background, the other
    fighter -- never reaches SAM2. The link is created here and removed again at
    bind time unless the shot supplies exclude_points (see app/comfy.py), so
    jobs without red points run the same graph as before.
    """
    editors = [key for key, node in prompt.items() if node["class_type"] == "PointsEditor"]
    segmenters = [key for key, node in prompt.items() if node["class_type"] == "Sam2Segmentation"]
    if not editors or not segmenters:
        return
    if len(editors) != 1:
        raise RuntimeError(f"expected one PointsEditor in the Animate template, found {len(editors)}")
    names = list(((object_info or {}).get("PointsEditor") or {}).get("output_name") or ["positive_coords", "negative_coords"])
    index = names.index("negative_coords")  # ValueError = the node's outputs were renamed: fail loudly
    for key in segmenters:
        prompt[key]["inputs"]["coordinates_negative"] = [editors[0], index]


ANIMAL_POSE_NODE_ID = "animal_pose_ap10k"


def _combo_options(spec) -> list:
    """Options of a combo input in /object_info, old (`[[a, b], {...}]`) or new
    (`["COMBO", {"options": [a, b]}]`) format."""
    if isinstance(spec, list) and spec:
        if isinstance(spec[0], list):
            return spec[0]
        if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
            return list(spec[1].get("options", []))
    return []


def add_animal_pose_branch(prompt: dict, object_info: dict | None) -> bool:
    """Adds comfyui_controlnet_aux's AnimalPosePreprocessor (AP10K, 17 quadruped
    keypoints) next to the human DWPose one, fed by the very same squared driving
    frames. Nothing consumes it until a shot asks for pose_source="animal" (see
    app/comfy.py), so human jobs run the unchanged, validated graph.

    Returns False (and adds nothing) when the installed node pack has no such node or no
    recognizable models.
    The node is a ControlNet preprocessor for SD1.5's animal-openpose; no source says
    Wan 2.2 Animate was trained on its skeleton, so the path is experimental.
    """
    spec = (object_info or {}).get("AnimalPosePreprocessor")
    if not spec:
        return False
    human = [
        key for key, node in prompt.items()
        if node["class_type"] == "DWPreprocessor" and node["inputs"].get("detect_body") == "enable"
    ]
    if len(human) != 1:
        raise RuntimeError(f"expected one body DWPreprocessor in the Animate template, found {len(human)}")
    # comfyui_controlnet_aux declares the detector / estimator under `optional` (confirmed on the
    # live /object_info 2026-10-03); accept either section.
    declared = spec.get("input") or {}
    fields = {**(declared.get("required") or {}), **(declared.get("optional") or {})}
    detectors = _combo_options(fields.get("bbox_detector"))
    estimators = _combo_options(fields.get("pose_estimator"))
    detector = next((o for o in ("yolox_l.onnx", "yolox_l.torchscript.pt") if o in detectors), None)
    estimator = next((o for o in ("rtmpose-m_ap10k_256_bs5.torchscript.pt", "rtmpose-m_ap10k_256.onnx") if o in estimators), None)
    if detector is None or estimator is None:
        # Never let the optional animal path take the validated human one down with it.
        print(
            "warning: AnimalPosePreprocessor has no known YOLOX detector / AP10K estimator "
            f"(detectors={detectors}, estimators={estimators}); pose_source='animal' is disabled",
            file=sys.stderr,
        )
        return False
    shared = prompt[human[0]]["inputs"]
    prompt[ANIMAL_POSE_NODE_ID] = {
        "class_type": "AnimalPosePreprocessor",
        "inputs": {
            "image": shared["image"],
            "bbox_detector": detector,
            "pose_estimator": estimator,
            "resolution": shared["resolution"],
        },
    }
    return True


def build_bindings(prompt: dict, workflow_name: str = "wan22_t2v") -> dict:
    bindings: dict[str, Any] = {}
    load_images: list[str] = []
    model_loaders: dict[str, str] = {}
    animate_width_targets: list[str] = []
    animate_height_targets: list[str] = []
    animate_background_inputs: list[str] = []
    sam2_negative_inputs: list[str] = []
    animate_pose_inputs: list[str] = []
    animate_face_inputs: list[str] = []
    for node_id, node in prompt.items():
        kind, inputs = node["class_type"], node["inputs"]
        title = node.get("_meta", {}).get("title", "").lower()
        if kind == "WanAnimateToVideo":
            # Wan 2.2 Animate's official template chains 3 WanAnimateToVideo
            # calls (one per Sampling/Extend subgraph instance), each with its
            # own width/height widget for the same clip -- bind all of them so
            # a requested resolution actually applies everywhere, not just
            # whichever instance happens to be processed last.
            if "width" in inputs:
                animate_width_targets.append(pointer(node_id, "width"))
            if "height" in inputs:
                animate_height_targets.append(pointer(node_id, "height"))
            # The background handling of "Mix": the SAM2 mask of the replaced
            # person and the driving video with that person blacked out. A
            # WanAnimateToVideo that does not receive them ("Move") generates the
            # scene from the reference image instead -- confirmed live 2026-10-02:
            # dropping both inputs from all three nodes gave the same character
            # in the reference's own dojo / rooftop / forest background.
            if "pose_video" in inputs:
                animate_pose_inputs.append(pointer(node_id, "pose_video"))
            if "face_video" in inputs:
                animate_face_inputs.append(pointer(node_id, "face_video"))
            for background_input in ("background_video", "character_mask"):
                if background_input in inputs:
                    animate_background_inputs.append(pointer(node_id, background_input))
        if kind == "LoadVideo" and "driving_video" not in bindings:
            # The official template ships this node with its "file" combo
            # unset (no widgets_values), so the converter emits it with an
            # empty inputs dict -- pointer() still targets the right place;
            # _json_pointer_set adds the "file" key when the real filename is
            # bound. ComfyUI requires "file" to be present at all (see
            # object_info: LoadVideo.required.file), so this is mandatory,
            # not optional like the video-mode-specific fields below.
            bindings["driving_video"] = pointer(node_id, "file")
        if kind == "CLIPTextEncode" and "text" in inputs:
            bindings["negative_prompt" if "negative" in title else "positive_prompt"] = pointer(node_id, "text")
        if kind == "RandomNoise" and "noise_seed" in inputs and "seed" not in bindings:
            bindings["seed"] = pointer(node_id, "noise_seed")
        if kind in {
            "EmptyHunyuanLatentVideo",
            "Wan22ImageToVideoLatent",
            "WanImageToVideo",
            "WanFirstLastFrameToVideo",
        }:
            for source, target in (("width", "width"), ("height", "height"), ("length", "frames")):
                if source in inputs:
                    bindings[target] = pointer(node_id, source)
        if kind == "BasicScheduler" and "steps" in inputs:
            bindings["steps"] = pointer(node_id, "steps")
        if kind == "CFGGuider" and "cfg" in inputs:
            bindings["cfg"] = pointer(node_id, "cfg")
        if kind == "CreateVideo" and "fps" in inputs:
            bindings["fps"] = pointer(node_id, "fps")
        if kind in {"SaveVideo", "VHS_VideoCombine"}:
            for name in ("filename_prefix",):
                if name in inputs:
                    bindings["output_prefix"] = pointer(node_id, name)
        if kind == "LoadImage" and "image" in inputs:
            load_images.append(node_id)
        if kind == "Sam2Segmentation" and "coordinates_negative" in inputs:
            sam2_negative_inputs.append(pointer(node_id, "coordinates_negative"))
        if kind == "PointsEditor":
            # The template's SAM2 click-point, meant for a human to set in
            # ComfyUI's web UI before running the graph -- see
            # app/sam2_seed_point.py for why a headless caller must supply
            # its own computed point instead of leaving the template's
            # hardcoded (256,256) default (confirmed broken 2026-10-01: SAM2
            # seeds from a point that doesn't land on the subject, producing
            # a near pass-through of the driving video instead of the new
            # character).
            bindings["sam2_points_store"] = pointer(node_id, "points_store")
            bindings["sam2_coordinates"] = pointer(node_id, "coordinates")
            bindings["sam2_neg_coordinates"] = pointer(node_id, "neg_coordinates")
            editor_w, editor_h = inputs.get("width"), inputs.get("height")
            if (editor_w, editor_h) != (640, 640):
                raise RuntimeError(
                    "PointsEditor canvas changed from the pinned 640x640 "
                    f"(now {editor_w}x{editor_h}) -- app/sam2_seed_point.py's "
                    "coordinate transform assumes 640x640 and must be updated "
                    "to match before this binding can be trusted"
                )
        if kind in {"UNETLoader", "CheckpointLoaderSimple"}:
            filename = str(inputs.get("unet_name", inputs.get("ckpt_name", ""))).lower()
            model_loaders[node_id] = filename
    if load_images:
        bindings["start_image"] = pointer(load_images[0], "image")
    if len(load_images) > 1:
        bindings["end_image"] = pointer(load_images[1], "image")
    steps_binding, cfg_binding, sampler_seed = sampling_bindings(prompt)
    if steps_binding:
        bindings["steps"] = steps_binding
    if cfg_binding:
        bindings["cfg"] = cfg_binding
    if sampler_seed and "seed" not in bindings:
        bindings["seed"] = sampler_seed
    for loader_id, filename in model_loaders.items():
        branch = "high" if "high_noise" in filename else "low" if "low_noise" in filename else "main"
        candidates = [
            consumer_id
            for consumer_id, consumer in prompt.items()
            if consumer["class_type"] == "ModelSamplingSD3"
            and model_depends_on(prompt, consumer["inputs"].get("model"), loader_id)
        ]
        if len(candidates) == 1:
            bindings[f"{branch}_model_target"] = pointer(candidates[0], "model")
    if workflow_name == "wan22_animate":
        # Single unified diffusion model (no high/low-noise split), so the branch
        # heuristic in this function falls back to "main_model_target".
        # "keep_background" (Mix vs Move) is not a node in this official template,
        # so it is bound as "background_inputs": the pointers to every
        # WanAnimateToVideo background_video/character_mask input. bind_workflow
        # deletes them for ANIMATE_MOVE (keep_background=False), which is what
        # makes Move render the reference image's own background.
        #
        # sam2_points_store/sam2_coordinates/sam2_neg_coordinates ARE now
        # wired (see the PointsEditor branch above + app/sam2_seed_point.py):
        # the caller must compute them from the driving video rather than
        # leave the template's hardcoded (256,256) default, which doesn't
        # reliably land on the subject.
        if animate_background_inputs:
            bindings["background_inputs"] = animate_background_inputs
        if sam2_negative_inputs:
            bindings["sam2_negative_input"] = sam2_negative_inputs
        if ANIMAL_POSE_NODE_ID in prompt and animate_pose_inputs:
            # pose_source="animal": every stage's pose_video comes from the AP10K
            # skeleton instead of DWPose, and the human face crops are dropped.
            bindings["pose_video_inputs"] = animate_pose_inputs
            bindings["face_video_inputs"] = animate_face_inputs
            bindings["animal_pose_output"] = [ANIMAL_POSE_NODE_ID, 0]
        if animate_width_targets:
            bindings["width"] = animate_width_targets
        if animate_height_targets:
            bindings["height"] = animate_height_targets
        # No "frames" binding on purpose: each of the 3 chained WanAnimateToVideo
        # calls has its own "length" (an internal per-segment sample-window size,
        # confirmed 77 on the official template), not a single "total output
        # frames" knob -- the template's Sampling -> Extend -> Extend chain fixes
        # the overall clip length architecturally. Making the requested
        # shot.frames actually control total length needs someone to trace how
        # video_frame_offset/TrimVideoLatent stitch the 3 segments together, not
        # attempted here. Until then, an animate shot's `frames` field is
        # silently ignored (see _apply_bound_value: a missing binding is a
        # no-op) and the template's own fixed length is used.
        required = {
            "positive_prompt",
            "seed",
            "width",
            "height",
            "fps",
            "output_prefix",
            "main_model_target",
            "start_image",
            "driving_video",
            "sam2_points_store",
            "sam2_coordinates",
            "sam2_neg_coordinates",
            "background_inputs",
            "sam2_negative_input",
        }
    else:
        required = {
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
        }
        if workflow_name == "wan22_i2v":
            required.add("start_image")
        elif workflow_name == "wan22_flf2v":
            required.update({"start_image", "end_image"})
    missing = required - set(bindings)
    if missing:
        raise RuntimeError(f"automatic binding failed, missing {sorted(missing)}")
    return bindings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    info = get_json(f"{COMFY_URL}/object_info")
    DEST.mkdir(parents=True, exist_ok=True)
    for name, source_name in FILES.items():
        destination = DEST / f"{name}.api.json"
        source_path = SOURCE / source_name
        try:
            source_hash = source_sha256(source_path)
            if not args.force and reusable(destination, source_hash):
                print(f"valid: {destination}")
                continue
            workflow = json.loads(source_path.read_text(encoding="utf-8"))
            prompt, turbo = convert(workflow, info, include_metadata=True)
            if name == "wan22_animate":
                wire_sam2_negative_points(prompt, info)
                add_animal_pose_branch(prompt, info)
            bindings = build_bindings(prompt, name)
            validate_runtime_prompt(prompt, name)
        except Exception as exc:
            if name not in EXPERIMENTAL:
                raise
            print(
                f"skipped (experimental, not yet converting cleanly): {name}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        prompt["_bindings"] = bindings
        if turbo is not None:
            prompt["_turbo"] = turbo
        prompt["_source_sha256"] = source_hash
        prompt["_converter_schema_version"] = CONVERTER_SCHEMA_VERSION
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=DEST, delete=False, suffix=".tmp"
        ) as handle:
            json.dump(prompt, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, destination)
        print(f"created: {destination}")


if __name__ == "__main__":
    main()
