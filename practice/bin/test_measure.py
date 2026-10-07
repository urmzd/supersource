"""Behavior checks for the benchmark probe, including Linux RSS inheritance."""

from pathlib import Path
import subprocess
import sys
import unittest


MEASURE = Path(__file__).with_name("measure.py")


class MeasureTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "linux", "Linux inherits pre-exec RSS")
    def test_parent_allocation_is_not_charged_to_native_child(self):
        launcher = """
import runpy
import sys
allocation = bytearray(64 * 1024 * 1024)
sys.argv = [sys.argv[1], "2", "/bin/true"]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
        result = subprocess.run(
            [sys.executable, "-c", launcher, str(MEASURE)],
            capture_output=True,
            text=True,
            check=True,
        )
        wall_ms, peak_kb = map(int, result.stdout.split())
        self.assertGreaterEqual(wall_ms, 0)
        self.assertLess(peak_kb, 16 * 1024)

    def test_real_child_allocation_is_measured_without_revealing_output(self):
        result = subprocess.run(
            [
                sys.executable,
                str(MEASURE),
                "2",
                sys.executable,
                "-c",
                "allocation = bytearray(32 * 1024 * 1024); print('hidden output')",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        wall_ms, peak_kb = map(int, result.stdout.split())
        self.assertGreaterEqual(wall_ms, 0)
        self.assertGreaterEqual(peak_kb, 32 * 1024)
        self.assertEqual(result.stderr, "")

    def test_failed_command_is_not_reported_as_a_measurement(self):
        result = subprocess.run(
            [sys.executable, str(MEASURE), "1", sys.executable, "-c", "exit(7)"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn("exited 7", result.stderr)


if __name__ == "__main__":
    unittest.main()
