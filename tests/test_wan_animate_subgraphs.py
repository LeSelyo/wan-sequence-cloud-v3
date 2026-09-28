import json
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import prepare_workflows  # noqa: E402

FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures/wan22_animate/video_wan2_2_14B_animate.json"
)


class ExpandChainedSubgraphsTests(unittest.TestCase):
    """Wan 2.2 Animate's official template chains 3 subgraph instances
    (Sampling -> Extend -> Extend), where a later instance's image1/
    video_frame_offset inputs are wired directly from the previous
    instance's own outputs. expand_subgraphs() used to raise
    "a subgraph input connected from another subgraph is unsupported"
    on this shape; it must now flatten it correctly instead."""

    def setUp(self):
        self.workflow = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_flattens_without_raising_and_has_no_dangling_links(self):
        flat = prepare_workflows.expand_subgraphs(self.workflow)
        node_ids = {str(node["id"]) for node in flat["nodes"]}
        for link in flat["links"]:
            self.assertIn(str(link["origin_id"]), node_ids, link)
            self.assertIn(str(link["target_id"]), node_ids, link)

    def test_chained_instance_input_resolves_to_real_producer_node(self):
        flat = prepare_workflows.expand_subgraphs(self.workflow)
        nodes_by_id = {str(node["id"]): node for node in flat["nodes"]}
        # instance 278 (second "Video Extend") takes image1 from instance
        # 242 (first "Video Extend")'s own IMAGE output -- both instances'
        # types are subgraph-instance placeholders, so the flattened link's
        # origin must be a real node type, never the bare instance id.
        cross_chain_links = [
            link
            for link in flat["links"]
            if str(link["target_id"]).startswith("subgraph:278:")
            and str(link["origin_id"]).startswith("subgraph:242:")
        ]
        self.assertTrue(cross_chain_links, "expected at least one 242->278 link")
        for link in cross_chain_links:
            origin_node = nodes_by_id[str(link["origin_id"])]
            self.assertNotIn("subgraph", str(origin_node.get("type", "")).lower())


if __name__ == "__main__":
    unittest.main()
