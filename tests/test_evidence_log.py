"""Tests for the opt-in JSONL evidence logger.

Run from the project root:

    python3 tests/test_evidence_log.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import evidence_log  # noqa: E402


class EvidenceLogTest(unittest.TestCase):
    def tearDown(self) -> None:
        evidence_log.close()

    def test_disabled_by_default_is_noop(self) -> None:
        evidence_log.close()
        self.assertFalse(evidence_log.is_enabled())
        evidence_log.log("translate", seq=1, jp="x", vi="y")  # must not raise

    def test_writes_jsonl_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.jsonl"
            self.assertIsNotNone(evidence_log.configure(str(path)))
            self.assertTrue(evidence_log.is_enabled())
            evidence_log.log("enqueue", seq=1, text="本日", queue_size=0)
            evidence_log.log("translate", seq=1, jp="本日", vi="Hôm nay", batch=1)
            evidence_log.close()

            lines = path.read_text(encoding="utf-8").splitlines()
            events = [json.loads(line) for line in lines]
            names = [e["event"] for e in events]
            self.assertIn("session_start", names)
            self.assertIn("enqueue", names)
            self.assertIn("translate", names)
            translate = next(e for e in events if e["event"] == "translate")
            self.assertEqual(translate["seq"], 1)
            self.assertEqual(translate["vi"], "Hôm nay")
            # Correlation fields are always present.
            for key in ("ts", "t_ms", "thread", "event"):
                self.assertIn(key, translate)

    def test_unexpected_types_are_serialized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.jsonl"
            evidence_log.configure(str(path))
            evidence_log.log("path_event", payload=Path("meeting.log"))
            evidence_log.close()

            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            event = next(e for e in events if e["event"] == "path_event")
            self.assertEqual(event["payload"], "meeting.log")

    def test_dropped_event_counter_surfaces_logger_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.jsonl"
            evidence_log.configure(str(path))
            before = evidence_log.get_dropped_event_count()
            circular: list[object] = []
            circular.append(circular)

            evidence_log.log("bad_event", payload=circular)

            self.assertEqual(evidence_log.get_dropped_event_count(), before + 1)
            self.assertEqual(evidence_log.summarize()["dropped_events"], before + 1)

    def test_configure_none_disables(self) -> None:
        self.assertIsNone(evidence_log.configure(None))
        self.assertFalse(evidence_log.is_enabled())


if __name__ == "__main__":
    raise SystemExit(unittest.main(verbosity=2))
