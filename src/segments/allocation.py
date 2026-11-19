"""有限随访名额的分配。

约束（任一不满足都不得硬派）：
* 联系授权：``no`` 不进入外呼名单；``limited`` 标注受限渠道；
* 能力适配：工作人员技能与居民组匹配（general 兜底）；
* 常跑范围：只派给 site_range 覆盖居民站点的工作人员；
* 名额与承诺期限：每人 weekly_slots 封顶，deadline_days 越短越优先；
* 不重复打扰：同一居民单轮最多 maximum_repeat_contacts 条建议，
  已在执行链路中的居民不再产生新建议；
* 同住限制：家庭关系未确认/已撤回时不得合并预约，confirmed 时也
  只允许在各自授权范围内分别联系；
* 保底：reserved_share 的名额预留给欠覆盖片区（无工作人员常跑或
  可及机会覆盖低的站点），保底用不满要在缺口报告里说明缺什么；
* 已接受服务不因模型换标签改约；已执行事项保留当时依据。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .clustering import SegmentationResult
from .features import FeatureTable
from .identity import HouseholdRegistry
from .models import (
    Action,
    ActionState,
    ContactPermission,
    Resident,
    Segment,
    Worker,
)


@dataclass
class AllocationCandidate:
    resident: str
    segment: Segment
    site: str
    group: str
    contact_permission: ContactPermission
    urgency: int                 # 越大越优先
    under_served: bool
    reason_code: str
    blocked: tuple[str, ...]     # 无法分配的具体原因（缺什么）


@dataclass
class GapItem:
    area: str
    missing: str
    detail: str


@dataclass
class AllocationPlan:
    run_id: str
    assignments: list[Action]
    candidates: list[AllocationCandidate]
    gaps: list[GapItem]
    reserved_target: int
    reserved_filled: int
    capacity_total: int
    capacity_used: int
    by_worker: dict[str, int]
    by_site: dict[str, int]
    by_segment: dict[str, int]
    restricted_contacts: tuple[str, ...]

    @property
    def reserved_shortfall(self) -> int:
        return max(0, self.reserved_target - self.reserved_filled)


# 分群 -> 建议的处置原因码与紧急度（机会受限优先走供给侧，外呼紧急度低）。
SEGMENT_POLICY = {
    Segment.SHORT_BREAK: ("reengage_confirm_reason", 3),
    Segment.OPPORTUNITY_CONSTRAINED: ("supply_side_adjustment", 1),
    Segment.SUSTAINED: ("retain_no_outbound", 0),
    Segment.UNCLASSIFIED: ("collect_evidence_first", 0),
}


def _skill_matches(worker: Worker, group: str) -> bool:
    return worker.skill == "general" or worker.skill == group


def build_candidates(
    residents: dict[str, Resident],
    segmentation: SegmentationResult,
    feature_table: FeatureTable,
    registry: HouseholdRegistry,
    staffed_sites: set[str],
    existing_actions: dict[str, Action],
) -> list[AllocationCandidate]:
    features = feature_table.by_resident()
    candidates: list[AllocationCandidate] = []
    for rid, resident in residents.items():
        seg = segmentation.segments[rid]
        reason_code, urgency = SEGMENT_POLICY[seg]
        blocked: list[str] = []

        if rid in existing_actions:
            state = existing_actions[rid].state
            if state in (ActionState.ACCEPTED, ActionState.PENDING, ActionState.COMPLETED):
                blocked.append(f"already_{state.value}")
        if not resident.contactable:
            blocked.append("contact_permission_no")
        if urgency == 0:
            blocked.append(f"segment_{seg.value}_not_actionable")
        link = registry.link(resident.household)
        if link.state.value == "withdrawn":
            blocked.append("household_authorization_withdrawn")

        cov = features[rid].values["opportunity_coverage"]
        under_served = (
            resident.site not in staffed_sites
            or cov < 0.5
        )
        candidates.append(AllocationCandidate(
            resident=rid, segment=seg, site=resident.site, group=resident.group,
            contact_permission=resident.contact_permission,
            urgency=urgency, under_served=under_served,
            reason_code=reason_code, blocked=tuple(blocked),
        ))
    return candidates


def allocate(
    run_id: str,
    residents: dict[str, Resident],
    workers: dict[str, Worker],
    segmentation: SegmentationResult,
    feature_table: FeatureTable,
    registry: HouseholdRegistry,
    *,
    reserved_share: float,
    maximum_repeat_contacts: int,
    existing_actions: Optional[dict[str, Action]] = None,
) -> AllocationPlan:
    existing_actions = dict(existing_actions or {})
    workers = dict(workers)
    staffed_sites = {w.site_range for w in workers.values()}
    candidates = build_candidates(
        residents, segmentation, feature_table, registry,
        staffed_sites, existing_actions,
    )

    capacity_total = sum(w.weekly_slots for w in workers.values())
    reserved_target = round(capacity_total * reserved_share)
    remaining = {w.worker: w.weekly_slots for w in workers.values()}
    by_worker = {w.worker: 0 for w in workers.values()}
    by_site: dict[str, int] = {}
    by_segment: dict[str, int] = {}
    gaps: list[GapItem] = []
    assignments: list[Action] = []
    restricted: list[str] = []
    contact_count: dict[str, int] = {}
    reserved_filled = 0

    def try_assign(cand: AllocationCandidate) -> bool:
        nonlocal reserved_filled
        eligible_workers = [
            w for w in workers.values()
            if remaining[w.worker] > 0
            and w.site_range == cand.site
            and _skill_matches(w, cand.group)
        ]
        if not eligible_workers:
            why = []
            if not any(w.site_range == cand.site for w in workers.values()):
                why.append("该站点无工作人员常跑")
            elif not any(_skill_matches(w, cand.group)
                         for w in workers.values() if w.site_range == cand.site):
                why.append(f"该站点无 {cand.group} 适配技能的工作人员")
            else:
                why.append("适配工作人员名额已满")
            gaps.append(GapItem(area=cand.site, missing="worker_capacity",
                                detail=f"{cand.resident}: " + "；".join(why)))
            return False
        # 承诺期限短的优先（deadline_days 小），保证能守住回访承诺。
        eligible_workers.sort(key=lambda w: (w.deadline_days, w.worker))
        worker = eligible_workers[0]
        action = Action(
            action_id=f"{run_id}-{cand.resident}",
            resident=cand.resident, worker=worker.worker,
            segment=cand.segment, reason_code=cand.reason_code,
            state=ActionState.RECOMMENDED, created_run=run_id,
            basis={
                "segment": cand.segment.value,
                "site": cand.site, "group": cand.group,
                "under_served": cand.under_served,
                "contact_permission": cand.contact_permission.value,
                "alternative_access": feature_table.by_resident()[
                    cand.resident
                ].values["alternative_access"],
                "segmentation_version": segmentation.version,
            },
        )
        assignments.append(action)
        remaining[worker.worker] -= 1
        by_worker[worker.worker] += 1
        by_site[cand.site] = by_site.get(cand.site, 0) + 1
        by_segment[cand.segment.value] = by_segment.get(cand.segment.value, 0) + 1
        contact_count[cand.resident] = contact_count.get(cand.resident, 0) + 1
        if cand.contact_permission is ContactPermission.LIMITED:
            restricted.append(cand.resident)
        if cand.under_served:
            reserved_filled += 1
        return True

    # 第一阶段：欠覆盖片区保底，按紧急度排序，但不把名额全堆给容易联系的片区。
    under = [
        c for c in candidates
        if c.under_served and not c.blocked
        and contact_count.get(c.resident, 0) < maximum_repeat_contacts
    ]
    under.sort(key=lambda c: (-c.urgency, c.site, c.resident))
    reserved_seen: set[str] = set()
    for cand in under:
        if reserved_filled >= reserved_target:
            break
        if cand.resident in reserved_seen:
            continue
        reserved_seen.add(cand.resident)
        try_assign(cand)

    # 第二阶段：普通分配，仍受单对象外呼上限约束。
    rest = [
        c for c in candidates
        if not c.under_served and not c.blocked
        and contact_count.get(c.resident, 0) < maximum_repeat_contacts
    ]
    rest.sort(key=lambda c: (-c.urgency, c.site, c.resident))
    for cand in rest:
        if sum(remaining.values()) == 0:
            gaps.append(GapItem(area="global", missing="capacity",
                                detail="全部工作人员名额已用尽，仍有可分配候选未排上"))
            break
        try_assign(cand)

    # 第三阶段：把无法分配的候选转化为"缺什么"的显式缺口。
    for cand in candidates:
        if cand.blocked:
            gaps.append(GapItem(
                area=cand.site, missing="eligibility",
                detail=f"{cand.resident}: {','.join(cand.blocked)}",
            ))
        elif not any(a.resident == cand.resident for a in assignments):
            if not any(g.area == cand.site and g.detail.startswith(cand.resident)
                       for g in gaps):
                gaps.append(GapItem(
                    area=cand.site, missing="unallocated",
                    detail=f"{cand.resident}: 符合条件但本轮名额不足",
                ))

    if reserved_filled < reserved_target:
        gaps.append(GapItem(
            area="reserved_quota",
            missing="under_served_candidates_or_capacity",
            detail=(
                f"欠覆盖片区保底名额目标 {reserved_target}，实际填入 {reserved_filled}，"
                "缺口原因：欠覆盖片区候选不足或缺少对应站点/技能的工作人员"
            ),
        ))

    return AllocationPlan(
        run_id=run_id, assignments=assignments, candidates=candidates, gaps=gaps,
        reserved_target=reserved_target, reserved_filled=reserved_filled,
        capacity_total=capacity_total, capacity_used=len(assignments),
        by_worker=by_worker, by_site=by_site, by_segment=by_segment,
        restricted_contacts=tuple(restricted),
    )
