"""
Unit tests for MCP server tool endpoints.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlx_commander.mcp_server import (
    convert_dataset_headless_tool,
    estimate_fine_tuning_resources_tool,
    estimate_test_eval_throughput_tool,
    execute_queue_tool,
    inspect_dataset_tool,
    inspect_queue_tool,
    launch_conversion_tui_tool,
    launch_lora_tui_tool,
    queue_multi_run_sweep_tool,
    queue_single_run_tool,
    run_mcp_server,
    run_test_evaluation_tool,
)
from tests.conftest import make_sample_qa_records


class TestMcpServer(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.src_file = Path(self.temp_dir) / "source.jsonl"
        self.out_dir = Path(self.temp_dir) / "mlx_output"

        records = make_sample_qa_records(20)
        with open(self.src_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_inspect_dataset_tool(self):
        result = inspect_dataset_tool(str(self.src_file))
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["total_rows"], 20)
        self.assertIn("instruction", result["columns"])
        self.assertIn("output", result["columns"])
        self.assertIn("suggested_mappings", result)
        self.assertIn("prompt_completion", result["suggested_mappings"])

    def test_convert_dataset_headless_tool(self):
        result = convert_dataset_headless_tool(
            dataset_path=str(self.src_file),
            format="prompt_completion",
            prompt_col="instruction",
            completion_col="output",
            train_pct=80.0,
            valid_pct=20.0,
            test_pct=0.0,
            output_dir=str(self.out_dir),
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["total_records"], 20)
        self.assertEqual(result["splits"]["train"], 16)
        self.assertEqual(result["splits"]["valid"], 4)
        self.assertTrue(Path(result["output_dir"]).exists())

    @patch("mlx_commander.mcp_server.is_macos", return_value=True)
    @patch("mlx_commander.mcp_server.spawn_terminal_tui")
    def test_launch_conversion_tui_tool(self, mock_spawn, mock_is_mac):
        mock_spawn.return_value = 0
        manifest_path = self.out_dir / "mlx_manifest.json"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"status": "success", "total_records": 20}, f)

        res = launch_conversion_tui_tool(
            dataset_path=str(self.src_file),
            format="prompt_completion",
            output_dir=str(self.out_dir),
        )

        mock_spawn.assert_called_once()
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["total_records"], 20)

    @patch("mlx_commander.mcp_server.is_macos", return_value=False)
    def test_launch_conversion_tui_tool_non_macos(self, mock_is_mac):
        res = launch_conversion_tui_tool(
            dataset_path=str(self.src_file),
            format="prompt_completion",
            output_dir=str(self.out_dir),
        )
        self.assertEqual(res["status"], "error")
        self.assertIn("supported on macOS", res["message"])

    def test_run_mcp_server_missing_dep(self):
        # In environment without mcp installed, run_mcp_server exits with 1
        with patch.dict("sys.modules", {"mcp": None, "mcp.server.fastmcp": None}):
            ec = run_mcp_server()
            self.assertEqual(ec, 1)

    def test_estimate_fine_tuning_resources_tool(self):
        res = estimate_fine_tuning_resources_tool(
            model="mlx-community/Llama-3.2-3B-Instruct-4bit",
            iters=500,
            batch_size=4,
            gradient_accumulation_steps=1,
            total_train_records=1000,
        )
        self.assertEqual(res["status"], "success")
        self.assertIn("peak_memory_gb", res)
        self.assertIn("safety_tier", res)
        self.assertIn("implied_epochs", res)
        self.assertAlmostEqual(res["implied_epochs"], 2.0)
        self.assertIn("estimated_duration_str", res)

    def test_queue_single_run_tool_standard_and_vlm(self):
        queue_d = Path(self.temp_dir) / "test_queue_single"
        # Standard LM run
        res = queue_single_run_tool(
            model="mlx-community/Llama-3.2-3B-Instruct-4bit",
            data_path=str(self.out_dir),
            name="test_standard_run",
            batch_size=4,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            iters=100,
            queue_dir=str(queue_d),
        )
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["batch_size"], 4)
        self.assertEqual(res["gradient_accumulation_steps"], 1)
        self.assertFalse(res["vlm_safeguard_applied"])

        # VLM run with batch_size > 1 -> should trigger safeguard (batch=1, grad_accum=4)
        res_vlm = queue_single_run_tool(
            model="google/gemma-4-4b-it",
            data_path=str(self.out_dir),
            batch_size=4,
            gradient_accumulation_steps=1,
            engine="mlx_vlm",
            queue_dir=str(queue_d),
        )
        self.assertEqual(res_vlm["status"], "success")
        self.assertTrue(res_vlm["vlm_safeguard_applied"])
        self.assertEqual(res_vlm["batch_size"], 1)
        self.assertEqual(res_vlm["gradient_accumulation_steps"], 4)
        self.assertEqual(res_vlm["effective_batch_size"], 4)

    def test_queue_multi_run_sweep_tool(self):
        queue_d = Path(self.temp_dir) / "test_queue_sweep"
        # 2 learning rates * 2 ranks = 4 runs
        res = queue_multi_run_sweep_tool(
            model="mlx-community/Llama-3.2-3B-Instruct-4bit",
            data_path=str(self.out_dir),
            learning_rate=[1e-4, 2e-4],
            lora_rank=[8, 16],
            queue_dir=str(queue_d),
        )
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["total_runs"], 4)
        self.assertEqual(len(res["queued_runs"]), 4)
        self.assertIn("total_estimated_duration_str", res)

    def test_inspect_queue_tool(self):
        queue_d = Path(self.temp_dir) / "test_inspect_queue"
        queue_single_run_tool(
            model="mlx-community/Llama-3.2-3B-Instruct-4bit",
            data_path=str(self.out_dir),
            name="job_1",
            queue_dir=str(queue_d),
        )
        queue_single_run_tool(
            model="mlx-community/Llama-3.2-3B-Instruct-4bit",
            data_path=str(self.out_dir),
            name="job_2",
            queue_dir=str(queue_d),
        )

        res = inspect_queue_tool(queue_dir=str(queue_d))
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["total_runs"], 2)
        self.assertEqual(res["pending_count"], 2)
        self.assertEqual(len(res["runs"]), 2)
        self.assertEqual(res["runs"][0]["name"], "job_1")
        self.assertEqual(res["runs"][1]["name"], "job_2")

    @patch("mlx_commander.mcp_server.is_macos", return_value=True)
    @patch("mlx_commander.mcp_server.spawn_terminal_tui", return_value=0)
    def test_execute_queue_tool_spawn(self, mock_spawn, mock_is_mac):
        queue_d = Path(self.temp_dir) / "test_exec_queue"
        res = execute_queue_tool(queue_dir=str(queue_d), spawn_terminal=True)
        mock_spawn.assert_called_once()
        self.assertEqual(res["status"], "success")

    def test_estimate_test_eval_throughput_tool(self):
        res_small = estimate_test_eval_throughput_tool("mlx-community/Llama-3.2-1B-Instruct-4bit", total_samples=50)
        res_large = estimate_test_eval_throughput_tool("meta-llama/Llama-3.1-70B-Instruct-4bit", total_samples=50)
        self.assertEqual(res_small["status"], "success")
        self.assertEqual(res_large["status"], "success")
        self.assertGreater(res_small["est_tps"], res_large["est_tps"])
        self.assertLess(res_small["est_total_sec"], res_large["est_total_sec"])

    @patch("mlx_commander.mcp_server.run_generative_eval")
    def test_run_test_evaluation_tool(self, mock_run_eval):
        mock_run_eval.return_value = {
            "status": "success",
            "summary": {
                "exact_match_pct": 80.0,
                "substring_match_pct": 95.0,
                "avg_word_f1": 0.88,
            },
            "html_dashboard": "adapters/eval_comparison.html",
        }
        res = run_test_evaluation_tool(
            model="mlx-community/Llama-3.2-3B-Instruct-4bit",
            adapter_path="adapters/my_adapter",
            data_path=str(self.out_dir),
        )
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["summary"]["exact_match_pct"], 80.0)

    @patch("mlx_commander.mcp_server.is_macos", return_value=True)
    @patch("mlx_commander.mcp_server.spawn_terminal_tui", return_value=0)
    def test_launch_lora_tui_tool(self, mock_spawn, mock_is_mac):
        res = launch_lora_tui_tool(mode=2, dataset_path=str(self.out_dir))
        mock_spawn.assert_called_once()
        self.assertEqual(res["status"], "success")
        calls = mock_spawn.call_args[0][0]
        self.assertIn("--lora", calls)
        self.assertIn("--dataset", calls)

        mock_spawn.reset_mock()
        res3 = launch_lora_tui_tool(mode=3)
        self.assertEqual(res3["status"], "success")
        calls3 = mock_spawn.call_args[0][0]
        self.assertIn("--multi-run", calls3)


if __name__ == "__main__":
    unittest.main()
