import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "src", _ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from segments.allocation import allocate
from segments.clustering import segment
from segments.dataio import Dataset
from segments.features import OpportunityView, build_features
from segments.identity import HouseholdRegistry
from segments.models import (
    Action,
    ActionState,
    IdentityState,
    SEGMENTATION_VERSION,
    Segment,
)
from segments.run import run_pipeline
from segments.stability import (
    MATCH_RELIABILITY_FLOOR,
    compare_runs,
    held_out_site_stability,
    temporal_stability,
)

from helpers import build_dataset, four_week_opportunities


def _dataset(tmp: Path, policy=None) -> Dataset:
    return Dataset.load(_write(tmp, policy))


def _write(tmp: Path, policy=None) -> Path:
    # 两个站点各 6 名居民，覆盖三种联系授权与两个行为组
    residents = [
        ["a1", "h1", "s1", "adult", "yes", "pm"],
        ["a2", "h2", "s1", "adult", "limited", "pm"],
        ["a3", "h3", "s1", "adult", "no", "pm"],
        ["b1", "h4", "s2", "senior", "yes", "weekend"],
        ["b2", "h5", "s2", "senior", "limited", "weekend"],
        ["b3", "h6", "s2", "senior", "yes", "weekend"],
        # s3 无工作人员覆盖（欠覆盖片区）
        ["c1", "h7", "s3", "youth", "yes", "am"],
        ["c2", "h8", "s3", "youth", "yes", "am"],
    ]
    events = []
    for rid in ("a1", "a3"):                          # 持续出席
        for j in range(4):
            events.append([rid, f"c{j % 2}", f"w{j % 2}", "attend", "travel"])
    # a2 为受限授权且短期中断，验证受限渠道标注
    # c1/c2 位于无工作人员的 s3 且短期中断，验证欠覆盖片区缺口
    for rid in ("a2", "b1", "b2", "b3", "c1", "c2"):  # 短期中断（失约）
        for j in range(4):
            events.append([rid, f"c{j % 2}", f"w{j % 2}", "no-show", "unknown"])
    opp = four_week_opportunities(["s1", "s2", "s3"])
    staff = [
        ["w1", "general", "s1", 2, 7],
        ["w2", "senior", "s2", 2, 3],
    ]
    return build_dataset(Path(tmp), residents=residents, events=events,
                         opportunities=opp, staff=staff, policy=policy)


class StabilityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ds = Dataset.load(_write(Path(self.tmp.name)))
        self.reg = HouseholdRegistry(self.ds.residents)

    def _segment(self, anchor=7):
        view = OpportunityView.build(self.ds.opportunities, 8, anchor)
        table = build_features(self.ds.residents, self.ds.events, view,
                               self.reg, 8)
        return segment(table)

    def test_temporal_and_held_out_site_agreement(self):
        t = temporal_stability(self.ds.residents, self.ds.events,
                               self.ds.opportunities, self.reg, 8)
        self.assertGreaterEqual(t.agreement, MATCH_RELIABILITY_FLOOR)
        self.assertTrue(t.explanations_ok)
        h = held_out_site_stability(self.ds.residents, self.ds.events,
                                    self.ds.opportunities, self.reg, 8)
        self.assertGreaterEqual(h.agreement, MATCH_RELIABILITY_FLOOR)
        # 留站检查必须覆盖全部站点
        self.assertEqual(set(h.detail["site_agreement"]), {"s1", "s2", "s3"})

    def test_version_mismatch_terminates_comparison(self):
        a = self._segment()
        b = copy.deepcopy(a)
        object.__setattr__(b, "version", "v9.9")
        cmp = compare_runs("old", a, "new", b)
        self.assertFalse(cmp.comparable)
        self.assertTrue(cmp.definition_switch)
        self.assertIn("口径", cmp.terminated_reason)
        self.assertEqual(cmp.changed_residents(), {})

    def test_unreliable_match_terminates_comparison(self):
        import random
        a = self._segment()
        # 随机重排一半标签，制造无法一一对应的簇结构，使匹配质量下降
        rng = random.Random(7)
        b = copy.deepcopy(a)
        rids = list(b.labels)
        for rid in rng.sample(rids, len(rids) // 2):
            b.labels[rid] = rng.randrange(max(2, b.k))
        # 在更严格的运营口径下，低于门槛的匹配必须终止而不是硬出变化名单
        cmp = compare_runs("run-a", a, "run-b", b, reliability_floor=0.8)
        self.assertEqual(cmp.version_a, SEGMENTATION_VERSION)
        self.assertFalse(cmp.comparable)
        self.assertIsNotNone(cmp.terminated_reason)
        self.assertLess(cmp.match_quality, 0.8)

    def test_comparable_run_reports_behavior_changes(self):
        a = self._segment()
        b = self._segment()
        cmp = compare_runs("run-a", a, "run-b", b)
        self.assertTrue(cmp.comparable)
        self.assertGreaterEqual(cmp.match_quality, MATCH_RELIABILITY_FLOOR)
        self.assertIsInstance(cmp.changed_residents(), dict)


class AllocationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ds = Dataset.load(_write(Path(self.tmp.name)))
        self.reg = HouseholdRegistry(self.ds.residents)
        view = OpportunityView.build(self.ds.opportunities, 8)
        self.table = build_features(self.ds.residents, self.ds.events, view,
                                    self.reg, 8)
        self.seg = segment(self.table)

    def _allocate(self, existing=None, policy=None):
        policy = policy or self.ds.policy
        return allocate(
            "run-x", self.ds.residents, self.ds.workers, self.seg, self.table,
            self.reg, reserved_share=policy.reserved_share,
            maximum_repeat_contacts=policy.maximum_repeat_contacts,
            existing_actions=existing,
        )

    def test_respects_contact_permission_skill_range_and_capacity(self):
        plan = self._allocate()
        assigned = {a.resident: a for a in plan.assignments}
        # no 授权居民绝不入名单
        self.assertNotIn("a3", assigned)
        for action in plan.assignments:
            resident = self.ds.residents[action.resident]
            worker = self.ds.workers[action.worker]
            self.assertEqual(worker.site_range, resident.site)
            self.assertTrue(worker.skill in ("general", resident.group))
        # 名额不超上限
        for wid, used in plan.by_worker.items():
            self.assertLessEqual(used, self.ds.workers[wid].weekly_slots)
        # limited 授权被标注
        self.assertIn("a2", plan.restricted_contacts)

    def test_under_served_area_gap_is_reported_not_hidden(self):
        plan = self._allocate()
        # s3 无工作人员，候选必须出现在缺口里，说明缺什么
        s3_gaps = [g for g in plan.gaps if g.area == "s3"]
        self.assertTrue(s3_gaps)
        self.assertTrue(any(g.missing == "worker_capacity" for g in s3_gaps))
        # 保底缺口显式报告
        self.assertGreater(plan.reserved_shortfall, 0)
        self.assertTrue(any(g.area == "reserved_quota" for g in plan.gaps))

    def test_accepted_service_is_locked_across_runs(self):
        first = self._allocate()
        action = next(a for a in first.assignments if a.resident == "b1")
        action.transition(ActionState.ACCEPTED, "worker", "居民接受")
        existing = {a.resident: a for a in first.assignments}
        second = self._allocate(existing=existing)
        # 已接受的动作原样保留，不生成新建议、不改派工作人员
        self.assertNotIn("b1", {a.resident for a in second.assignments})
        self.assertIs(existing["b1"].state, ActionState.ACCEPTED)
        self.assertEqual(existing["b1"].history[-1]["to"], "accepted")

    def test_no_duplicate_outbound_for_same_resident(self):
        first = self._allocate()
        existing = {a.resident: a for a in first.assignments}
        second = self._allocate(existing=existing)
        # 同一轮/链路中每个居民最多一条建议
        residents = [a.resident for a in second.assignments]
        self.assertEqual(len(residents), len(set(residents)))

    def test_withdrawn_household_authorization_blocks_new_contact(self):
        # b1/b2 属于不同家庭；让 b1 的家庭关系撤回（虽然这里是单人家庭，
        # 改用多档案夹具验证撤回语义）
        residents = [
            ["m1", "hh", "s1", "adult", "yes", "pm"],
            ["m2", "hh", "s2", "senior", "yes", "weekend"],
        ]
        events = [["m1", "c0", "w0", "no-show", "unknown"],
                  ["m2", "c1", "w1", "no-show", "unknown"]]
        with TemporaryDirectory() as raw:
            path = build_dataset(
                Path(raw), residents=residents, events=events,
                opportunities=four_week_opportunities(["s1", "s2"]),
                staff=[["w1", "general", "s1", 3, 7],
                       ["w2", "senior", "s2", 3, 3]])
            ds = Dataset.load(path)
        reg = HouseholdRegistry(ds.residents)
        reg.withdraw("hh")
        view = OpportunityView.build(ds.opportunities, 8)
        table = build_features(ds.residents, ds.events, view, reg, 8)
        seg = segment(table)
        plan = allocate("run-y", ds.residents, ds.workers, seg, table, reg,
                        reserved_share=0.25, maximum_repeat_contacts=1)
        for candidate in plan.candidates:
            if candidate.resident in ("m1", "m2"):
                self.assertIn("household_authorization_withdrawn",
                              candidate.blocked)

    def test_sustained_residents_get_no_outbound(self):
        plan = self._allocate()
        for action in plan.assignments:
            self.assertIsNot(action.segment, Segment.SUSTAINED)


if __name__ == "__main__":
    unittest.main()
