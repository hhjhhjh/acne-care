import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from acne_care.evaluation.runtime import build_model_metadata, synchronize_torch_device


class EvaluationRuntimeTest(unittest.TestCase):
    def test_synchronizes_numbered_cuda_device(self) -> None:
        cuda_synchronize = Mock()
        torch_runtime = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, synchronize=cuda_synchronize),
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: False)
            ),
            mps=SimpleNamespace(synchronize=Mock()),
        )

        synchronize_torch_device("0", torch_runtime)

        cuda_synchronize.assert_called_once_with()

    def test_does_not_synchronize_cuda_for_cpu(self) -> None:
        cuda_synchronize = Mock()
        torch_runtime = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, synchronize=cuda_synchronize),
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: False)
            ),
            mps=SimpleNamespace(synchronize=Mock()),
        )

        synchronize_torch_device("cpu", torch_runtime)

        cuda_synchronize.assert_not_called()

    def test_builds_model_metadata_with_explicit_id_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            weights = Path(temp_dir) / "best.pt"
            weights.write_bytes(b"weights")

            metadata = build_model_metadata("dataset-v2-control", weights)

        self.assertEqual(metadata["id"], "dataset-v2-control")
        self.assertEqual(metadata["weights"], str(weights))
        self.assertEqual(
            metadata["weights_sha256"],
            "9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c",
        )

    def test_rejects_blank_model_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            weights = Path(temp_dir) / "best.pt"
            weights.write_bytes(b"weights")

            with self.assertRaisesRegex(ValueError, "model id"):
                build_model_metadata("   ", weights)


if __name__ == "__main__":
    unittest.main()
