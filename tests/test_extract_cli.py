import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ExtractTier1CliTest(unittest.TestCase):
    def test_help_exposes_separate_toolace_and_lmcache_sources(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "extract_tier1_labels.py"), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("toolace", result.stdout)
        self.assertIn("lmcache", result.stdout)
        self.assertIn("--output", result.stdout)


if __name__ == "__main__":
    unittest.main()
