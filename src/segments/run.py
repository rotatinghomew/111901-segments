"""端到端流水线与命令行入口。

一次分群运行（run）是不可变的分析产物：分群口径版本、特征、
簇画像、稳定性报告与分配计划都挂在同一个 run_id 下。运营调整
分群时，新 run 与旧 run 通过 :func:`compare_runs` 比较，从而区分
"居民行为真的变了"与"分群口径切换"。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .allocation import AllocationPlan, allocate
from .clustering import SegmentationResult, segment
from .dataio import Dataset
from .features import OpportunityView, build_features
from .identity import HouseholdRegistry, reconcile
from .models import Action, Segment
from .review import WeeklyReview, build_weekly_review
from .stability import StabilityReport, compare_runs, run_stability_report


@dataclass
class SegmentationRun:
    run_id: str
    anchor_week: int
    dataset_path: str
    segmentation: SegmentationResult
    stability: Optional[StabilityReport] = None
    allocation: Optional[AllocationPlan] = None
    review: Optional[WeeklyReview] = None
    registry_snapshot: dict = field(default_factory=dict)

    def report(self) -> dict:
        seg = self.segmentation
        out = {
            "run_id": self.run_id,
            "segmentation_version": seg.version,
            "anchor_week": self.anchor_week,
            "silhouette": round(seg.silhouette, 4),
            "support_floor": seg.support_floor,
            "segment_counts": {k.value: v for k, v in seg.segment_counts().items()},
            "profiles": [
                {
                    "cluster": p.cluster_id,
                    "segment": p.segment.value,
                    "size": p.size,
                    "discriminators": [
                        {"feature": f, "z": round(z, 3), "direction": d}
                        for f, z, d in p.discriminators
                    ],
                    "rationale": p.naming_rationale,
                }
                for p in seg.profiles
            ],
            "unclassified": {
                rid: reason.value for rid, reason in seg.unclassified_reasons.items()
            },
            "stability": (
                [
                    {
                        "name": c.name,
                        "agreement": round(c.agreement, 4),
                        "coverage": round(c.coverage, 4),
                        "unclassified_rate": round(c.unclassified_rate, 4),
                        "explanations_ok": c.explanations_ok,
                        "detail": c.detail,
                    }
                    for c in self.stability.checks
                ]
                if self.stability else None
            ),
            "allocation": _allocation_dict(self.allocation) if self.allocation else None,
            "review": self.review.as_dict() if self.review else None,
            "identity_snapshot": self.registry_snapshot,
            "notes": seg.notes,
        }
        return out


def _allocation_dict(plan: AllocationPlan) -> dict:
    return {
        "capacity_total": plan.capacity_total,
        "capacity_used": plan.capacity_used,
        "reserved_target": plan.reserved_target,
        "reserved_filled": plan.reserved_filled,
        "reserved_shortfall": plan.reserved_shortfall,
        "by_worker": plan.by_worker,
        "by_site": plan.by_site,
        "by_segment": plan.by_segment,
        "restricted_contacts": list(plan.restricted_contacts),
        "assignments": [
            {
                "action_id": a.action_id,
                "resident": a.resident,
                "worker": a.worker,
                "segment": a.segment.value,
                "reason_code": a.reason_code,
                "state": a.state.value,
                "basis": a.basis,
            }
            for a in plan.assignments
        ],
        "gaps": [asdict(g) for g in plan.gaps],
    }


def run_pipeline(
    data_dir: str | Path,
    run_id: str,
    *,
    anchor_week: Optional[int] = None,
    registry: Optional[HouseholdRegistry] = None,
    existing_actions: Optional[dict[str, Action]] = None,
    with_stability: bool = True,
) -> tuple[SegmentationRun, Dataset, HouseholdRegistry, SegmentationResult, dict[str, Action]]:
    dataset = Dataset.load(data_dir)
    if registry is None:
        registry = HouseholdRegistry(dataset.residents)
    reconciliation = reconcile(dataset.residents, dataset.events, registry)
    if not reconciliation.clean:
        raise ValueError(
            "身份校对未通过，未知居民标识: "
            + ",".join(reconciliation.unknown_event_residents)
        )

    weeks = sorted(int(w[1:]) for w in dataset.weeks)
    anchor = anchor_week if anchor_week is not None else max(weeks)
    view = OpportunityView.build(
        dataset.opportunities, dataset.policy.lookback_weeks, anchor
    )
    table = build_features(
        dataset.residents, dataset.events, view, registry,
        dataset.policy.lookback_weeks,
    )
    pending = {
        rid for rid, r in dataset.residents.items()
        if registry.state_of(r.household).value == "pending"
    }
    result = segment(table, pending_identity_residents=pending)

    run = SegmentationRun(
        run_id=run_id, anchor_week=anchor, dataset_path=str(data_dir),
        segmentation=result,
        registry_snapshot={
            hh: link.state.value for hh, link in registry.households.items()
        },
    )
    if with_stability:
        run.stability = run_stability_report(
            dataset.residents, dataset.events, dataset.opportunities,
            registry, dataset.policy.lookback_weeks, pending=pending,
        )

    plan = allocate(
        run_id, dataset.residents, dataset.workers, result, table, registry,
        reserved_share=dataset.policy.reserved_share,
        maximum_repeat_contacts=dataset.policy.maximum_repeat_contacts,
        existing_actions=existing_actions,
    )
    run.allocation = plan

    actions = dict(existing_actions or {})
    for action in plan.assignments:
        actions[action.resident] = action
    return run, dataset, registry, result, actions


def add_weekly_review(
    run: SegmentationRun,
    dataset: Dataset,
    *,
    before_window: tuple[int, int],
    after_window: tuple[int, int],
    actions: dict[str, Action],
) -> WeeklyReview:
    review = build_weekly_review(
        actions=actions, events=dataset.events,
        opportunities=dataset.opportunities,
        before_window=before_window, after_window=after_window,
        segments_at_run={
            rid: seg.value for rid, seg in run.segmentation.segments.items()
        },
    )
    run.review = review
    return review


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="参与分群与随访名额分配流水线")
    parser.add_argument("--data", default="data", help="数据目录")
    parser.add_argument("--run-id", default="run-0001")
    parser.add_argument("--anchor-week", type=int, default=None)
    parser.add_argument("--out", default=None, help="报告输出 JSON 路径")
    args = parser.parse_args(argv)

    run, _, _, _, _ = run_pipeline(args.data, args.run_id, anchor_week=args.anchor_week)
    payload = json.dumps(run.report(), ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
