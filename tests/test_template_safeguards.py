import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlx_commander.lora.template_safeguards import (
    dataset_has_thinking_traces,
    heal_chat_template_string,
    heal_model_chat_template,
    probe_chat_template,
)


LLAMA3_TEMPLATE = """{{- bos_token }}
{%- for message in messages %}
    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}
{%- endif %}"""


MINISTRAL_2410_BUGGY_TEMPLATE = """{%- if messages[0]["role"] == "system" %}
    {%- set system_message = messages[0]["content"] %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set loop_messages = messages %}
{%- endif %}
{{- bos_token }}
{%- for message in loop_messages %}
    {%- if message["role"] == "user" %}
        {%- if loop.last and system_message is defined %}
            {{- "[INST]" + system_message + "\\n\\n" + message["content"] + "[/INST]" }}
        {%- else %}
            {{- "[INST]" + message["content"] + "[/INST]" }}
        {%- endif %}
    {%- elif message["role"] == "assistant" %}
        {{- " " + message["content"]|trim + eos_token }}
    {%- endif %}
{%- endfor %}"""


QWEN3_THINKING_TEMPLATE = """{%- if messages[0].role == 'system' %}
    {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}
{%- endif %}
{%- set ns = namespace(last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if message.role == "user" %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.role == "user" %}
        {{- '<|im_start|>user\\n' + message.content + '<|im_end|>\\n' }}
    {%- elif message.role == "assistant" %}
        {%- if loop.index0 > ns.last_query_index and loop.last %}
            {{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' + message.content + '<|im_end|>\\n' }}
        {%- else %}
            {{- '<|im_start|>assistant\\n' + message.content + '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- endif %}
{%- endif %}"""


GEMMA_RAISING_TEMPLATE = """{%- if messages[0]['role'] == 'system' %}
    {{- raise_exception('System role not supported') }}
{%- endif %}
{%- for message in messages %}
    {{- '<start_of_turn>' + message['role'] + '\\n' + message['content'] + '<end_of_turn>\\n' }}
{%- endfor %}"""


class TestChatTemplateSafeguards(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_probe_clean_llama_template(self):
        res = probe_chat_template(LLAMA3_TEMPLATE)
        self.assertEqual(res["status"], "ok")
        self.assertTrue(res["sys_in_eval"])
        self.assertTrue(res["sys_in_train"])
        self.assertTrue(res["prefix_match"])
        self.assertFalse(res["mask_offset_overflow"])
        self.assertFalse(res["injects_empty_think_in_train"])

    def test_probe_detects_ministral_train_eval_asymmetry(self):
        res = probe_chat_template(MINISTRAL_2410_BUGGY_TEMPLATE)
        self.assertEqual(res["status"], "broken")
        self.assertTrue(res["sys_in_eval"])
        self.assertFalse(res["sys_in_train"])
        self.assertTrue(res["mask_offset_overflow"])
        self.assertFalse(res["prefix_match"])

    def test_probe_detects_qwen3_unmasked_think_injection(self):
        res = probe_chat_template(QWEN3_THINKING_TEMPLATE)
        self.assertEqual(res["status"], "broken")
        self.assertTrue(res["sys_in_eval"])
        self.assertTrue(res["sys_in_train"])
        self.assertFalse(res["prefix_match"])
        self.assertTrue(res["injects_empty_think_in_train"])
        self.assertTrue(res["nothink_fixes_prefix"])

    def test_probe_detects_raising_system_role(self):
        res = probe_chat_template(GEMMA_RAISING_TEMPLATE)
        self.assertEqual(res["status"], "raises")
        self.assertTrue(res["raises_on_system"])

    def test_heal_ministral_template_string(self):
        healed, fixes = heal_chat_template_string(MINISTRAL_2410_BUGGY_TEMPLATE)
        self.assertEqual(len(fixes), 1)
        self.assertIn("loop.first", healed)
        after = probe_chat_template(healed)
        self.assertEqual(after["status"], "ok")
        self.assertTrue(after["sys_in_eval"])
        self.assertTrue(after["sys_in_train"])
        self.assertTrue(after["prefix_match"])
        self.assertFalse(after["mask_offset_overflow"])

    def test_heal_qwen3_template_string_for_non_thinking_dataset(self):
        healed, fixes = heal_chat_template_string(QWEN3_THINKING_TEMPLATE, disable_thinking=True)
        self.assertEqual(len(fixes), 1)
        after = probe_chat_template(healed)
        self.assertEqual(after["status"], "ok")
        self.assertTrue(after["prefix_match"])

    def test_preserves_qwen3_thinking_when_disable_thinking_is_false(self):
        healed, fixes = heal_chat_template_string(QWEN3_THINKING_TEMPLATE, disable_thinking=False)
        self.assertEqual(len(fixes), 0)
        self.assertEqual(healed, QWEN3_THINKING_TEMPLATE)

    def test_dataset_has_thinking_traces(self):
        ds_plain = self.base_path / "ds_plain"
        ds_plain.mkdir()
        with open(ds_plain / "train.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "messages": [
                    {"role": "system", "content": "Classify."},
                    {"role": "user", "content": "Profit grew."},
                    {"role": "assistant", "content": "positive"},
                ]
            }) + "\n")
        self.assertFalse(dataset_has_thinking_traces(ds_plain))

        ds_think = self.base_path / "ds_think"
        ds_think.mkdir()
        with open(ds_think / "train.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "messages": [
                    {"role": "user", "content": "Solve 2+2"},
                    {"role": "assistant", "content": "<think>\n2+2 is 4\n</think>\n4"},
                ]
            }) + "\n")
        self.assertTrue(dataset_has_thinking_traces(ds_think))

    def test_heal_model_chat_template_updates_tokenizer_config_and_creates_backup(self):
        model_dir = self.base_path / "Ministral-8B"
        model_dir.mkdir()
        tok_cfg = model_dir / "tokenizer_config.json"
        tok_cfg.write_text(
            json.dumps({"bos_token": "<s>", "eos_token": "</s>", "chat_template": MINISTRAL_2410_BUGGY_TEMPLATE}),
            encoding="utf-8",
        )

        res = heal_model_chat_template(model_dir)
        self.assertTrue(res["healed"])
        self.assertEqual(len(res["fixes"]), 1)
        self.assertTrue((model_dir / "tokenizer_config.json.bak").exists())

        updated = json.loads(tok_cfg.read_text(encoding="utf-8"))
        self.assertEqual(probe_chat_template(updated["chat_template"])["status"], "ok")

        # Second call is idempotent (already healed)
        res2 = heal_model_chat_template(model_dir)
        self.assertFalse(res2["healed"])

    def test_heal_model_chat_template_respects_disable_env(self):
        model_dir = self.base_path / "Qwen3-8B"
        model_dir.mkdir()
        tok_cfg = model_dir / "tokenizer_config.json"
        tok_cfg.write_text(
            json.dumps({"chat_template": QWEN3_THINKING_TEMPLATE}),
            encoding="utf-8",
        )
        with patch.dict(os.environ, {"MLX_DISABLE_TEMPLATE_SAFEGUARD": "1"}):
            res = heal_model_chat_template(model_dir)
            self.assertFalse(res["healed"])
            self.assertFalse((model_dir / "tokenizer_config.json.bak").exists())


if __name__ == "__main__":
    unittest.main()
