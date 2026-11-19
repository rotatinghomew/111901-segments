import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "src", _ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from segments.clustering import (
    UnclassifiedReason,
    _support_floor,
    segment,
)
from segments.features import FEATURE_COLUMNS, OpportunityView, build_features
from segments.identity import HouseholdRegistry
from segments.models import Event, EventType, Segment
from segments.dataio import Dataset

from helpers import build_dataset, four_week_opportunities


def make_resident_block(prefix, site, group, permission, event_type, reason,
                        households=4):
    """生成 4 名行为同质的居民，每人 4 条事件，落在 w0/w1 两周。"""
    residents, events = [], []
    for i in range(households):
        rid = f"{prefix}{i}"
        residents.append([rid, f"{prefix}-h{i}", site, group, permission, "pm"])
        for j in range(4):
            events.append([rid, f"c{j % 2}", f"w{j % 2}", event_type, reason])
    return residents, events


class FeatureTest(unittest.TestCase):
    def _dataset(self, tmp: Path) -> Dataset:
        residents, events = make_resident_block(
            "a", "s1", "adult", "yes", "attend", "travel")
        opp = four_week_opportunities(["s1"])
        staff = [["w1", "general", "s1", 10, 7]]
        path = build_dataset(Path(tmp), residents=residents, events=events,
                             opportunities=opp, staff=staff)
        return Dataset.load(path)

    def test_features_have_fixed_columns_and_evidence(self):
        with TemporaryDirectory() as raw:
            ds = self._dataset(Path(raw))
        reg = HouseholdRegistry(ds.residents)
        view = OpportunityView.build(ds.opportunities, 8)
        table = build_features(ds.residents, ds.events, view, reg, 8)
        self.assertEqual(table.columns, FEATURE_COLUMNS)
        self.assertEqual(len(table.rows), 4)
        row = table.by_resident()["a0"]
        self.assertEqual(row.evidence["events"], 4)
        self.assertEqual(row.evidence["attend_count"], 4)
        self.assertGreater(row.values["attend_rate"], 0.0)
        self.assertEqual(row.values["closure_exposure"], 0.0)
        # 向量长度与列顺序稳定，供聚类与跨期对齐使用
        self.assertEqual(len(row.vector()), len(FEATURE_COLUMNS))

    def test_closure_counts_as_supply_pressure_not_disinterest(self):
        with TemporaryDirectory() as raw:
            ds = self._dataset(Path(raw))
        # 把 s1 的 c0 全部闭馆：闭馆暴露应上升，但出席率特征不受惩罚性改写
        for key, opp in list(ds.opportunities.items()):
            if key[1] == "c0":
                object.__setattr__(opp, "closed", True)
                object.__setattr__(opp, "available", False)
        reg = HouseholdRegistry(ds.residents)
        view = OpportunityView.build(ds.opportunities, 8)
        table = build_features(ds.residents, ds.events, view, reg, 8)
        row = table.by_resident()["a0"]
        self.assertGreater(row.values["closure_exposure"], 0.0)
        self.assertGreater(row.values["attend_rate"], 0.0)
        self.assertTrue(any("机会" in n for n in row.notes) or row.evidence)

    def test_missing_opportunity_records_are_flagged_not_invented(self):
        residents = [["x1", "hx", "s9", "adult", "yes", "pm"]]
        events = [["x1", "c0", "w0", "signup", "schedule"]]
        opp = four_week_opportunities(["s1"])  # s9 完全没有机会记录
        with TemporaryDirectory() as raw:
            path = build_dataset(Path(raw), residents=residents, events=events,
                                 opportunities=opp,
                                 staff=[["w1", "general", "s1", 2, 7]])
            ds = Dataset.load(path)
        reg = HouseholdRegistry(ds.residents)
        view = OpportunityView.build(ds.opportunities, 8)
        table = build_features(ds.residents, ds.events, view, reg, 8)
        row = table.by_resident()["x1"]
        self.assertIn("home_site_no_opportunity_records", row.missing_flags)


class ClusteringTest(unittest.TestCase):
    def _four_pattern_dataset(self, tmp: Path) -> Dataset:
        blocks = [
            ("at", "s1", "adult", "yes", "attend", "travel"),
            ("sg", "s2", "senior", "limited", "signup", "schedule"),
            ("cc", "s3", "youth", "yes", "cancel", "closed"),
            ("ns", "s4", "adult", "no", "no-show", "unknown"),
        ]
        residents, events = [], []
        for b in blocks:
            r, e = make_resident_block(*b)
            residents += r
            events += e
        opp = four_week_opportunities(["s1", "s2", "s3", "s4"])
        staff = [
            ["w1", "general", "s1", 10, 7],
            ["w2", "senior", "s2", 10, 3],
            ["w3", "youth", "s3", 10, 5],
            ["w4", "general", "s4", 10, 7],
        ]
        return Dataset.load(build_dataset(
            Path(tmp), residents=residents, events=events,
            opportunities=opp, staff=staff))

    def test_finds_four_interpretable_clusters_without_thresholds(self):
        with TemporaryDirectory() as raw:
            ds = self._four_pattern_dataset(Path(raw))
        reg = HouseholdRegistry(ds.residents)
        view = OpportunityView.build(ds.opportunities, 8)
        table = build_features(ds.residents, ds.events, view, reg, 8)
        result = segment(table)
        # k 由轮廓系数选出，没有任何"出席率<阈值"式入参
        self.assertEqual(result.k, 4)
        counts = result.segment_counts()
        self.assertEqual(counts[Segment.SUSTAINED], 4)
        self.assertEqual(counts[Segment.OPPORTUNITY_CONSTRAINED], 4)
        self.assertEqual(counts[Segment.SHORT_BREAK], 8)  # 报名未出席 + 失约
        self.assertEqual(counts[Segment.UNCLASSIFIED], 0)
        # 每个簇都有可解释判别特征
        for profile in result.profiles:
            self.assertGreaterEqual(len(profile.discriminators), 2)
            self.assertTrue(profile.naming_rationale)

    def test_cancel_group_is_opportunity_limited_not_churn(self):
        with TemporaryDirectory() as raw:
            ds = self._four_pattern_dataset(Path(raw))
        reg = HouseholdRegistry(ds.residents)
        view = OpportunityView.build(ds.opportunities, 8)
        table = build_features(ds.residents, ds.events, view, reg, 8)
        result = segment(table)
        for i in range(4):
            self.assertIs(result.segments[f"cc{i}"],
                          Segment.OPPORTUNITY_CONSTRAINED)
            self.assertNotIn(f"cc{i}", result.unclassified_reasons)

    def test_supply_cut_moves_whole_site_to_opportunity_limited(self):
        with TemporaryDirectory() as raw:
            ds = self._four_pattern_dataset(Path(raw))
        for key, opp in list(ds.opportunities.items()):
            if key[0] == "s1":
                object.__setattr__(opp, "closed", True)
                object.__setattr__(opp, "available", False)
        reg = HouseholdRegistry(ds.residents)
        view = OpportunityView.build(ds.opportunities, 8)
        table = build_features(ds.residents, ds.events, view, reg, 8)
        result = segment(table)
        for i in range(4):
            self.assertIs(result.segments[f"at{i}"],
                          Segment.OPPORTUNITY_CONSTRAINED,
                          "站点供给收缩不得把居民切到负面组或弃置")
        self.assertFalse(
            any(v is Segment.UNCLASSIFIED for v in result.segments.values()))

    def test_evidence_insufficient_resident_stays_unclassified(self):
        # 有站点机会记录但本人零事件：不是流失，是证据不足
        residents = [
            ["q1", "hq", "s1", "adult", "yes", "pm"],
            ["q2", "hq2", "s1", "adult", "yes", "pm"],
        ]
        events = [["q2", "c0", "w0", "attend", "travel"],
                  ["q2", "c1", "w1", "attend", "travel"]]
        opp = four_week_opportunities(["s1"])
        with TemporaryDirectory() as raw:
            path = build_dataset(
                Path(raw), residents=residents, events=events, opportunities=opp,
                staff=[["w1", "general", "s1", 5, 7]])
            ds = Dataset.load(path)
        reg = HouseholdRegistry(ds.residents)
        view = OpportunityView.build(ds.opportunities, 8)
        table = build_features(ds.residents, ds.events, view, reg, 8)
        result = segment(table)
        self.assertIs(result.segments["q1"], Segment.UNCLASSIFIED)
        self.assertIn("q1", result.unclassified_reasons)

    def test_small_cluster_falls_back_to_unclassified(self):
        self.assertEqual(_support_floor(60), 3)
        # 构造 5 人：4 同质 + 1 孤立模式，孤立者应因小簇回落
        residents, events = make_resident_block(
            "a", "s1", "adult", "yes", "attend", "travel", households=4)
        residents.append(["lone", "hl", "s1", "senior", "yes", "weekend"])
        for j in range(4):
            events.append(["lone", f"c{j % 2}", f"w{j % 2}", "no-show", "unknown"])
        opp = four_week_opportunities(["s1"])
        with TemporaryDirectory() as raw:
            path = build_dataset(
                Path(raw), residents=residents, events=events, opportunities=opp,
                staff=[["w1", "general", "s1", 10, 7]])
            ds = Dataset.load(path)
        reg = HouseholdRegistry(ds.residents)
        view = OpportunityView.build(ds.opportunities, 8)
        table = build_features(ds.residents, ds.events, view, reg, 8)
        result = segment(table, max_k=4)
        self.assertIs(result.segments["lone"], Segment.UNCLASSIFIED)
        self.assertIs(result.unclassified_reasons["lone"],
                      UnclassifiedReason.SMALL_CLUSTER)

    def test_pending_identity_still_classified_on_own_evidence_but_flagged(self):
        with TemporaryDirectory() as raw:
            ds = self._four_pattern_dataset(Path(raw))
        reg = HouseholdRegistry(ds.residents)
        pending = set(ds.residents)
        view = OpportunityView.build(ds.opportunities, 8)
        table = build_features(ds.residents, ds.events, view, reg, 8)
        result = segment(table, pending_identity_residents=pending)
        self.assertIn("at0", result.pending_identity)
        self.assertIsNot(result.segments["at0"], Segment.UNCLASSIFIED)


if __name__ == "__main__":
    unittest.main()
