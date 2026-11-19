"""健身参与人群分群与社区随访名额分配分析包。

公开入口按分析顺序组织：
加载资料 -> 身份校对 -> 特征 -> 无阈值分群 -> 稳定性检查 ->
随访名额分配 -> 周会复盘。
"""

from __future__ import annotations

from .allocation import AllocationPlan, allocate
from .clustering import SegmentationResult, segment
from .dataio import Dataset, Policy
from .features import FEATURE_COLUMNS, FeatureTable, OpportunityView, build_features
from .identity import HouseholdRegistry, ReconciliationReport, reconcile
from .models import (
    Action,
    ActionState,
    ContactPermission,
    Event,
    EventType,
    IdentityState,
    Opportunity,
    Resident,
    SEGMENTATION_VERSION,
    SEGMENT_SUPPORT,
    Segment,
    Worker,
)
from .review import WeeklyReview, build_weekly_review
from .stability import (
    MATCH_RELIABILITY_FLOOR,
    RunComparison,
    StabilityReport,
    compare_runs,
    run_stability_report,
)

__all__ = [
    "Action", "ActionState", "AllocationPlan", "ContactPermission", "Dataset",
    "Event", "EventType", "FEATURE_COLUMNS", "FeatureTable", "HouseholdRegistry",
    "IdentityState", "MATCH_RELIABILITY_FLOOR", "Opportunity",
    "OpportunityView", "Policy", "ReconciliationReport", "Resident",
    "RunComparison", "SEGMENTATION_VERSION", "SEGMENT_SUPPORT", "Segment",
    "SegmentationResult", "StabilityReport", "WeeklyReview", "Worker",
    "allocate", "build_features", "build_weekly_review", "compare_runs",
    "reconcile", "run_stability_report", "segment",
]
