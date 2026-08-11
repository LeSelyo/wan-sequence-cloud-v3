import unittest

from app.catalog import lora_entry, validate_lora_for_stage
from app.schemas import SequenceRequest


class ContractTests(unittest.TestCase):
    def test_example_contract(self):
        request = SequenceRequest.model_validate(
            {
                "id": "test",
                "shots": [
                    {
                        "id": "one",
                        "mode": "t2v",
                        "prompt": "a bird flies",
                        "loras": [
                            {"id": "pixel_gamegirl_wan22_t2v_high", "target": "high"}
                        ],
                    }
                ],
            }
        )
        self.assertEqual(request.shots[0].mode.value, "t2v")

    def test_sd15_lora_rejected_for_wan(self):
        with self.assertRaisesRegex(ValueError, "incompatible"):
            validate_lora_for_stage(lora_entry("animal_chubby_sd15"), "wan22_i2v", "auto")

    def test_non_lora_is_absent(self):
        with self.assertRaisesRegex(ValueError, "unknown"):
            lora_entry("animaika")

    def test_keyframe_mode_requires_end(self):
        with self.assertRaises(ValueError):
            SequenceRequest.model_validate(
                {
                    "id": "bad",
                    "shots": [
                        {
                            "id": "one",
                            "mode": "i(keyframe)2v",
                            "start_image": {"path": "a.png"},
                        }
                    ],
                }
            )


if __name__ == "__main__":
    unittest.main()
