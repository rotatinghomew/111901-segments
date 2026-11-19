import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "src", _ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from segments.models import Action, ActionState, Segment
from segments.review import build_weekly_review
from segments.run import add_weekly_review, run_pipeline
from segments.dataio import Dataset

from helpers import build_dataset, four_week_opportunities


class ReviewTest(unittest.TestCase):
    def _dataset(self, tmp: Path) -> Dataset:
        residents = [
            ["p1", "h1", "s1", "adult", "yes", "pm"],
            ["p2", "h2", "s1", "adult", "yes", "pm"],
        ]
        # p1 早期失约、后期出席（可被重新参与口径捕捉）；p2 后期自然出席
        events = [
            ["p1", "c0", "w0", "no-show", "unknown"],
            ["p1", "c0", "w1", "no-show", "unknown"],
            ["p1", "c0", "w4", "attend", "travel"],
            ["p1", "c1", "w5", "attend", "travel"],
            ["p2", "c0", "w4", "attend", "travel"],
            ["p2", "c1", "w5", "attend", "travel"],
        ]
        opp = four_week_opportunities(["s1"])
        staff = [["w1", "general", "s1", 5, 7]]
        return Dataset.load(build_dataset(
            Path(tmp), residents=residents, events=events,
            opportunities=opp, staff=staff))

    def test_reengaged_vs_spontaneous_and_no_attribution(self):
        with TemporaryDirectory() as raw:
            ds = self._dataset(Path(raw))
        action = Action(
            action_id="a-p1", resident="p1", worker="w1",
            segment=Segment.SHORT_BREAK, reason_code="reengage",
            state=ActionState.COMPLETED, created_run="run-x",
            basis={"segment": Segment.SHORT_BREAK.value},
        )
        review = build_weekly_review(
            actions={"p1": action}, events=ds.events,
            opportunities=ds.opportunities,
            before_window=(0, 3), after_window=(4, 7),
            segments_at_run={"p1": Segment.SHORT_BREAK.value},
        )
        self.assertEqual(review.contacted_residents, ("p1",))
        self.assertEqual(review.reengaged, ("p1",))
        # p2 没有被联系却出席：归为自然回归，不能算外呼成果
        self.assertEqual(review.spontaneous_returns, ("p2",))
        self.assertEqual(review.reengagement_rate_contacted, 1.0)
        self.assertIn("不能归因", review.attribution_notice)
        # 机会变化逐站点给出
        self.assertTrue(all(hasattr(c, "net_usable_change")
                            for c in review.opportunity_changes))

    def test_recommended_only_is_not_counted_as_contacted(self):
        with TemporaryDirectory() as raw:
            ds = self._dataset(Path(raw))
        action = Action(
            action_id="a-p1", resident="p1", worker="w1",
            segment=Segment.SHORT_BREAK, reason_code="reengage",
            state=ActionState.RECOMMENDED, created_run="run-x", basis={},
        )
        review = build_weekly_review(
            actions={"p1": action}, events=ds.events,
            opportunities=ds.opportunities,
            before_window=(0, 3), after_window=(4, 7), segments_at_run={},
        )
        self.assertEqual(review.contacted_residents, ())
        self.assertEqual(review.action_summary.recommended, 1)


class PipelineIntegrationTest(unittest.TestCase):
    def test_pipeline_runs_on_repo_fixture_and_report_is_complete(self):
        run, ds, reg, result, actions = run_pipeline(
            Path(__file__).resolve().parents[1] / "data", "run-fixture")
        report = run.report()
        self.assertEqual(report["segmentation_version"], result.version)
        # 夹具里四类纯净行为模式各 15 人
        counts = report["segment_counts"]
        self.assertEqual(counts[Segment.SUSTAINED.value], 15)
        self.assertEqual(counts[Segment.OPPORTUNITY_CONSTRAINED.value], 15)
        self.assertEqual(counts[Segment.SHORT_BREAK.value], 30)
        # 稳定性与解释
        for check in report["stability"]:
            self.assertGreaterEqual(check["agreement"], 0.99)
            self.assertTrue(check["explanations_ok"])
        # 分配不超总名额，s4 缺口被显式记录
        self.assertLessEqual(report["allocation"]["capacity_used"],
                             report["allocation"]["capacity_total"])
        gap_areas = {g["area"] for g in report["allocation"]["gaps"]}
        self.assertIn("s4", gap_areas)
        # 周会复盘可以挂接
        review = add_weekly_review(run, ds, before_window=(0, 3),
                                   after_window=(4, 7), actions=actions)
        self.assertIn("attribution_notice", review.as_dict())


if __name__ == "__main__":
    unittest.main()
