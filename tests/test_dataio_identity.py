import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "src", _ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from segments.dataio import Dataset, Policy
from segments.identity import HouseholdRegistry, reconcile
from segments.models import ContactPermission, IdentityState

from helpers import build_dataset, four_week_opportunities


def _minimal_dataset(tmp: Path, **policy) -> Path:
    residents = [
        ["r1", "h1", "s1", "adult", "yes", "pm"],
        ["r2", "h1", "s2", "senior", "limited", "weekend"],
        ["r3", "h2", "s1", "youth", "no", "am"],
    ]
    events = [
        ["r1", "c0", "w0", "attend", "travel"],
        ["r1", "c0", "w1", "attend", "travel"],
        ["r2", "c1", "w0", "cancel", "closed"],
        ["r3", "c0", "w1", "no-show", "unknown"],
    ]
    opp = four_week_opportunities(["s1", "s2"])
    staff = [["w1", "general", "s1", 4, 7], ["w2", "senior", "s2", 2, 3]]
    return build_dataset(tmp, residents=residents, events=events,
                         opportunities=opp, staff=staff, policy=policy or None)


class LoadingTest(unittest.TestCase):
    def test_loads_roundtrip_and_parses_enums(self):
        with TemporaryDirectory() as raw:
            ds = Dataset.load(_minimal_dataset(Path(raw)))
        self.assertEqual(len(ds.residents), 3)
        self.assertIs(ds.residents["r3"].contact_permission, ContactPermission.NO)
        self.assertFalse(ds.residents["r3"].contactable)
        self.assertTrue(ds.residents["r1"].contactable)
        self.assertEqual(ds.policy.reserved_share, 0.25)
        self.assertEqual(ds.policy.lookback_weeks, 8)
        # usable 同时受 available 与 closed 控制
        self.assertIn(("s1", "c0", "w0"), ds.opportunities)
        self.assertTrue(ds.opportunities[("s1", "c0", "w0")].usable)

    def test_unknown_resident_in_events_fails_hard(self):
        with TemporaryDirectory() as raw:
            path = _minimal_dataset(Path(raw))
            with (path / "events.csv").open("a", encoding="utf-8", newline="") as f:
                f.write("r9,c0,w0,attend,travel\n")
            with self.assertRaises(ValueError):
                Dataset.load(path)

    def test_policy_validation(self):
        with TemporaryDirectory() as raw:
            with self.assertRaises((ValueError, KeyError)):
                _minimal_dataset(Path(raw), reserved_share=1.5,
                                 lookback_weeks=8, maximum_repeat_contacts=1)
                Dataset.load(Path(raw))


class IdentityTest(unittest.TestCase):
    def setUp(self):
        with TemporaryDirectory() as raw:
            self.ds = Dataset.load(_minimal_dataset(Path(raw)))

    def test_multi_profile_household_starts_pending_and_blocks_merging(self):
        reg = HouseholdRegistry(self.ds.residents)
        report = reconcile(self.ds.residents, self.ds.events, reg)
        self.assertTrue(report.clean)
        self.assertEqual(report.multi_profile_households, {"h1": ("r1", "r2")})
        self.assertIn("h1", report.pending_links)
        # 待确认时不允许共享授权/合并预约
        self.assertEqual(reg.authorized_household_contacts("r1"), ())
        part = reg.household_participation(self.ds.events)
        self.assertEqual(part["h1"].get("_unlinked"), 1)

    def test_confirm_then_withdraw_residency_state_machine(self):
        reg = HouseholdRegistry(self.ds.residents)
        reg.confirm("h1")
        self.assertIs(reg.state_of("h1"), IdentityState.CONFIRMED)
        self.assertEqual(reg.authorized_household_contacts("r1"), ("r2",))
        # 确认后家庭证据合并，出席+报名可见
        part = reg.household_participation(self.ds.events)
        self.assertEqual(part["h1"]["attend"], 2)
        reg.withdraw("h1")
        self.assertEqual(reg.authorized_household_contacts("r1"), ())
        self.assertTrue(any(h["to"] == "withdrawn" for h in reg.link("h1").history))

    def test_single_profile_household_is_confirmed_vacuously(self):
        reg = HouseholdRegistry(self.ds.residents)
        self.assertIs(reg.state_of("h2"), IdentityState.CONFIRMED)


if __name__ == "__main__":
    unittest.main()
