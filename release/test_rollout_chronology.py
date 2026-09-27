"""Synthetic settlement chronology; never a live rollback or timing policy."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from release.rollout import RehearsalJournal, RolloutRejected
from release.test_rollout import evidence, fixture_plan, observation


class RolloutChronologyTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = str(Path(self.temp.name) / "chronology.sqlite")
        self.plan = fixture_plan()
        self.store = RehearsalJournal(self.path, initialize=True)
        self.addCleanup(lambda: self.store.close())
        self.store.create(self.plan, observation(self.plan), now_ms=1100)
        self.intent = self.store.promote(0, observation(self.plan), evidence(self.plan, 0), now_ms=1500)

    def settlement(self, observed_at):
        value = observation(self.plan, {"core": 10, "ultra": 0}, observed_at=observed_at)
        value["settled_intent_id"] = self.intent["id"]
        return value

    def test_observation_before_pending_intent_cannot_authorize_rollback(self):
        before = self.store.status()
        for timestamp in (1, 1000, 1499):
            with self.subTest(timestamp=timestamp), self.assertRaises(RolloutRejected):
                self.store.rollback(1, self.settlement(timestamp), now_ms=1600)
            self.assertEqual(self.store.status(), before)

    def test_restart_preserves_the_original_settlement_time_boundary(self):
        self.store.close()
        self.store = RehearsalJournal(self.path)
        with self.assertRaises(RolloutRejected):
            self.store.rollback(1, self.settlement(1499), now_ms=1600)
        self.assertEqual(self.store.status()["pending"]["id"], self.intent["id"])
        self.assertEqual(self.store.status()["version"], 1)

    def test_matching_same_millisecond_observation_needs_no_artificial_delay(self):
        rollback = self.store.rollback(1, self.settlement(1500), now_ms=1600)
        self.assertEqual(rollback["action"], "rollback")
        state = self.store.confirm(2, rollback["id"],
            observation(self.plan, 0, intent_id=rollback["id"], observed_at=1700), now_ms=1800)
        self.assertTrue(state["rolled_back"])
        self.assertEqual(state["confirmed_weight"], 0)
        self.assertFalse(state["production_traffic_changed"])

    def test_corrupt_pending_timestamp_is_rejected_without_coercion_or_replacement(self):
        original = self.store._read()[1]
        for invalid in (None, True, "1500", 1500.0):
            state = deepcopy(original)
            state["pending"]["requested_at_ms"] = invalid
            encoded = json.dumps(state)
            self.store.db.execute("UPDATE rehearsal SET state=? WHERE id=1", (encoded,))
            with self.subTest(invalid=invalid), self.assertRaises(RolloutRejected):
                self.store.rollback(1, self.settlement(1500), now_ms=1600)
            self.assertEqual(self.store.db.execute("SELECT state FROM rehearsal WHERE id=1").fetchone()[0], encoded)


if __name__ == "__main__":
    unittest.main()
