"""Tests for pxx_run script"""
import os
import sys
import tempfile
import pickle
import unittest
from unittest.mock import patch, MagicMock

from polycrystalx.scripts import run


class DummyJob:
    def __init__(self, name="dummy"):
        self.name = name


class DummyModule:
    job = DummyJob("single_job")
    custom_job = DummyJob("custom_job")
    key_job = ("matl", "poly", "mesh", "defm")
    job_keys = ["key1", "key2", "key3"]
    custom_suite = ["k_a", "k_b"]

    @staticmethod
    def get_job(key):
        return DummyJob(f"job_for_{key}")


class TestPxxRunParser(unittest.TestCase):

    def setUp(self):
        self.parser = run.argparser()

    def test_single_job_default(self):
        args = self.parser.parse_args(["my_module"])
        self.assertEqual(args.input_module, "my_module")
        self.assertIsNone(args.attribute)
        self.assertFalse(args.suite)
        self.assertEqual(args.n, 2)
        attr = run.resolve_attribute(args)
        self.assertEqual(attr, "job")

    def test_single_job_custom_attribute(self):
        args = self.parser.parse_args(["my_module", "my_custom_job"])
        self.assertEqual(args.input_module, "my_module")
        self.assertEqual(args.attribute, "my_custom_job")
        self.assertFalse(args.suite)
        self.assertEqual(args.n, 2)
        attr = run.resolve_attribute(args)
        self.assertEqual(attr, "my_custom_job")

    def test_suite_default_attribute(self):
        args = self.parser.parse_args(["-s", "my_module"])
        self.assertEqual(args.input_module, "my_module")
        self.assertIsNone(args.attribute)
        self.assertTrue(args.suite)
        self.assertEqual(args.n, 2)
        attr = run.resolve_attribute(args)
        self.assertEqual(attr, "job_keys")

    def test_suite_custom_attribute(self):
        args = self.parser.parse_args(["-s", "my_module", "vary_bcs", "-n", "4"])
        self.assertEqual(args.input_module, "my_module")
        self.assertEqual(args.attribute, "vary_bcs")
        self.assertTrue(args.suite)
        self.assertEqual(args.n, 4)
        attr = run.resolve_attribute(args)
        self.assertEqual(attr, "vary_bcs")

    def test_suite_long_option(self):
        args = self.parser.parse_args(["--suite", "my_module"])
        self.assertTrue(args.suite)
        self.assertEqual(run.resolve_attribute(args), "job_keys")


class TestPxxRunExecution(unittest.TestCase):

    @patch("polycrystalx.scripts.run.get_input_module")
    @patch("polycrystalx.scripts.run.subprocess.run")
    def test_single_job_spawns_mpirun(self, mock_run, mock_get_module):
        mock_get_module.return_value = DummyModule
        mock_run.return_value = MagicMock(returncode=0)

        with patch.object(sys, "argv", ["pxx_run", "-n", "4", "demo_module"]):
            run.main()

        self.assertEqual(mock_run.call_count, 1)
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[0], "mpirun")
        self.assertEqual(cmd[1], "-n")
        self.assertEqual(cmd[2], "4")
        self.assertIn("--worker", cmd)
        self.assertIn("demo_module", cmd)
        self.assertIn("job", cmd)

    @patch("polycrystalx.scripts.run.get_input_module")
    @patch("polycrystalx.scripts.run.subprocess.run")
    def test_suite_spawns_mpirun_per_job(self, mock_run, mock_get_module):
        mock_get_module.return_value = DummyModule
        mock_run.return_value = MagicMock(returncode=0)

        with patch.object(sys, "argv", ["pxx_run", "-s", "-n", "3", "demo_module", "custom_suite"]):
            run.main()

        # custom_suite has 2 items ("k_a", "k_b")
        self.assertEqual(mock_run.call_count, 2)
        cmd = mock_run.call_args_list[0][0][0]
        self.assertEqual(cmd[0], "mpirun")
        self.assertEqual(cmd[1], "-n")
        self.assertEqual(cmd[2], "3")
        self.assertIn("--worker", cmd)
        self.assertIn("--key-file", cmd)
        self.assertIn("demo_module", cmd)
        self.assertIn("custom_suite", cmd)

    @patch("polycrystalx.scripts.run.get_input_module")
    def test_single_job_missing_attribute_raises(self, mock_get_module):
        mock_get_module.return_value = DummyModule

        with patch.object(sys, "argv", ["pxx_run", "demo_module", "nonexistent"]):
            with self.assertRaises(AttributeError):
                run.main()

    @patch("polycrystalx.scripts.run.get_input_module")
    def test_suite_missing_attribute_raises(self, mock_get_module):
        mock_get_module.return_value = DummyModule

        with patch.object(sys, "argv", ["pxx_run", "-s", "demo_module", "nonexistent"]):
            with self.assertRaises(AttributeError):
                run.main()

    @patch("polycrystalx.scripts.run.processes.run")
    @patch("polycrystalx.scripts.run.get_input_module")
    def test_worker_single_job(self, mock_get_module, mock_proc_run):
        mock_get_module.return_value = DummyModule

        with patch.object(sys, "argv", ["pxx_run", "--worker", "demo_module", "custom_job"]):
            run.main()

        mock_proc_run.assert_called_once_with(DummyModule.custom_job)

    @patch("polycrystalx.scripts.run.processes.run")
    @patch("polycrystalx.scripts.run.get_input_module")
    def test_worker_single_job_with_key_tuple(self, mock_get_module, mock_proc_run):
        mock_get_module.return_value = DummyModule

        with patch.object(sys, "argv", ["pxx_run", "--worker", "demo_module", "key_job"]):
            run.main()

        self.assertEqual(mock_proc_run.call_count, 1)
        job_arg = mock_proc_run.call_args[0][0]
        self.assertEqual(job_arg.name, "job_for_('matl', 'poly', 'mesh', 'defm')")

    @patch("polycrystalx.scripts.run.processes.run")
    @patch("polycrystalx.scripts.run.get_input_module")
    def test_worker_suite_job_with_key_file(self, mock_get_module, mock_proc_run):
        mock_get_module.return_value = DummyModule

        with tempfile.NamedTemporaryFile(delete=False) as fp:
            pickle.dump("test_key", fp)
            temp_path = fp.name

        try:
            with patch.object(sys, "argv", ["pxx_run", "--worker", "--key-file", temp_path, "demo_module", "job_keys"]):
                run.main()

            self.assertEqual(mock_proc_run.call_count, 1)
            job_arg = mock_proc_run.call_args[0][0]
            self.assertEqual(job_arg.name, "job_for_test_key")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main()
