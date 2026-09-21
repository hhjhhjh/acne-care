import hashlib
import tempfile
import unittest
from pathlib import Path

from acne_care.legacy.model_artifact import LegacyModelArtifact, verify_legacy_model


class LegacyModelArtifactTest(unittest.TestCase):
    def test_accepts_matching_file(self) -> None:
        content = b"legacy-weights"
        expected_hash = hashlib.sha256(content).hexdigest()

        with tempfile.TemporaryDirectory() as temp_dir:
            weight_path = Path(temp_dir) / "weights.h5"
            weight_path.write_bytes(content)
            artifact = LegacyModelArtifact(
                path=weight_path,
                size_bytes=len(content),
                sha256=expected_hash,
            )

            verify_legacy_model(artifact)

    def test_rejects_wrong_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            weight_path = Path(temp_dir) / "weights.h5"
            weight_path.write_bytes(b"different")
            artifact = LegacyModelArtifact(
                path=weight_path,
                size_bytes=len(b"different"),
                sha256="0" * 64,
            )

            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_legacy_model(artifact)

    def test_rejects_missing_file(self) -> None:
        artifact = LegacyModelArtifact(
            path=Path("missing.weights.h5"),
            size_bytes=1,
            sha256="0" * 64,
        )

        with self.assertRaisesRegex(FileNotFoundError, "missing.weights.h5"):
            verify_legacy_model(artifact)


if __name__ == "__main__":
    unittest.main()

