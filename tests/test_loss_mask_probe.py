import io
import unittest
from unittest.mock import patch

from mlx_commander.lora.runner import (
    probe_mlx_lm_loss_mask,
    warn_if_loss_mask_is_off_by_one,
)


class _FakeNtoks:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


def _fake_loss(graded):
    """Stand in for mlx-lm's default_loss, reporting a fixed number of graded tokens."""
    def _loss(model, batch, lengths):
        return None, _FakeNtoks(graded)
    return _loss


class TestLossMaskProbe(unittest.TestCase):
    """The probe reads how many positions mlx-lm grades; 3 is right for its batch."""

    def test_reports_one_extra_position_when_bug_present(self):
        self.assertEqual(probe_mlx_lm_loss_mask(loss_fn=_fake_loss(4)), 1)

    def test_reports_no_extra_position_when_fixed(self):
        self.assertEqual(probe_mlx_lm_loss_mask(loss_fn=_fake_loss(3)), 0)

    def test_returns_none_when_count_is_unrecognised(self):
        # Something else changed upstream; say "unknown" rather than guess.
        self.assertIsNone(probe_mlx_lm_loss_mask(loss_fn=_fake_loss(11)))

    def test_returns_none_when_probe_raises(self):
        def _boom(model, batch, lengths):
            raise RuntimeError("internals moved")
        self.assertIsNone(probe_mlx_lm_loss_mask(loss_fn=_boom))

    def test_warns_only_when_bug_is_present(self):
        for extra, expect_warning in ((1, True), (0, False), (None, False)):
            with self.subTest(extra=extra):
                buf = io.StringIO()
                with patch("mlx_commander.lora.runner.probe_mlx_lm_loss_mask", return_value=extra), \
                        patch("sys.stdout", buf):
                    warn_if_loss_mask_is_off_by_one()
                self.assertEqual("[Loss]" in buf.getvalue(), expect_warning)


if __name__ == "__main__":
    unittest.main()
