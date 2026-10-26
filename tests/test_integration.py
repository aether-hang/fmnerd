"""End-to-end smoke tests for the TM / DM / DC entry points.

They use the built-in ``Synthetic`` dataset and stop before evaluation, so
they exercise the complete training path (data loading, Stage 1, DDiF loss,
auxiliary losses, diagnostics, saving) in a few seconds on CPU.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(HERE)


def run_command(args):
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable] + args,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=300,
    )
    return result


class TestEntryPoints(unittest.TestCase):
    def _run(self, method, extra):
        with tempfile.TemporaryDirectory() as tmp:
            buffer_path = os.path.join(tmp, "buffers")
            logs = os.path.join(tmp, "logs")
            if method == "TM":
                make = run_command(
                    [os.path.join(HERE, "make_fake_buffer.py"),
                     "--buffer_path", buffer_path, "--model", "ConvNetD1"]
                )
                self.assertEqual(make.returncode, 0, make.stdout)
            common = [
                "--dataset", "Synthetic",
                "--model", "ConvNetD1",
                "--ipc", "1",
                "--Iteration", "1",
                "--eval_it", "1000",
                "--batch_syn", "2",
                "--msz_per_channel", "4",
                "--layer_size", "4",
                "--stage1_steps", "5",
                "--stage1_eval_every", "5",
                "--save_path", logs,
                "--mod_every", "1",
                "--freq_diag_every", "1",
                "--share_diag_every", "1",
                "--grad_every", "1",
                "--log_every", "1",
            ]
            if method == "TM":
                common += [
                    "--buffer_path", buffer_path,
                    "--syn_steps", "1",
                    "--expert_epochs", "1",
                    "--max_start_epoch", "2",
                ]
            result = run_command(extra + common)
            self.assertEqual(result.returncode, 0, result.stdout[-4000:])
            log_files = []
            for root, _, files in os.walk(logs):
                if "log.txt" in files:
                    log_files.append(os.path.join(root, "log.txt"))
            self.assertTrue(log_files, result.stdout)
            with open(log_files[0], encoding="utf-8") as handle:
                text = handle.read()
            for needle in [
                "[Stage 1]",
                "[Stage 2]",
                "grad norms",
                "modulation magnitude",
                "top-k frequency alignment residual",
                "shared backbone vs instance modulation",
            ]:
                self.assertIn(needle, text)
            return text

    def test_dm(self):
        self._run("DM", [os.path.join("DM", "main_DM_FMNeRD.py")])

    def test_tm(self):
        self._run("TM", [os.path.join("TM", "main_TM_FMNeRD.py")])

    def test_dc(self):
        self._run("DC", [os.path.join("DC", "main_DC_FMNeRD.py")])


if __name__ == "__main__":
    unittest.main()
