"""机会可及性与参与行为特征。

设计原则：
* 闭馆（closed=1）与满员（available=0 且未闭馆）是**供给侧**事实，
  只能削弱居民"有机会"的证据，绝不能作为居民失去兴趣的证据；
* 可选记录缺失（某站点某周没有课程行）记为"未知"，参与率类指标
  只在有可及机会的周上计算，并单独输出证据计数；
* 输出每个特征的证据量，供分群层决定是否归入"证据不足"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .identity import HouseholdRegistry
from .models import Event, EventType, Opportunity, Resident, week_index


# 特征列顺序固定，聚类与解释都按该顺序取值。
FEATURE_COLUMNS: tuple[str, ...] = (
    "attend_rate",               # 出席 / 可及机会周
    "signup_rate",               # 报名 / 可及机会周
    "cancel_rate",               # 取消 / 可及机会周
    "no_show_rate",              # 失约 / 可及机会周
    "opportunity_coverage",      # 可及机会周 / 回看周数
    "closure_exposure",          # 闭馆机会数 / 全部机会数
    "fullness_exposure",         # 满员机会数 / 全部机会数
    "mean_attend_gap_weeks",     # 相邻出席的平均间隔（无出席记 0，另给证据位）
    "recent_trend",              # 近半窗出席率 - 远半窗出席率
    "alternative_access",        # 可跨站替代的课程周占比
    "preference_concentration",  # 课程偏好集中度（课程占比平方和）
    "household_activity_ratio",  # 已确认家庭的合并参与率（未确认则为本人值）
)


@dataclass
class OpportunityView:
    """机会表的索引视图。"""

    by_key: dict[tuple[str, str, str], Opportunity]
    site_weeks: dict[str, dict[int, list[Opportunity]]]
    week_nos: tuple[int, ...]

    @classmethod
    def build(
        cls,
        opportunities: dict[tuple[str, str, str], Opportunity],
        lookback_weeks: int,
        anchor_week: Optional[int] = None,
    ) -> "OpportunityView":
        all_weeks = sorted({week_index(w) for (_, _, w) in opportunities})
        if anchor_week is None:
            anchor_week = max(all_weeks) if all_weeks else lookback_weeks - 1
        window = set(range(anchor_week - lookback_weeks + 1, anchor_week + 1))
        site_weeks: dict[str, dict[int, list[Opportunity]]] = {}
        for (site, course, wk), opp in opportunities.items():
            n = week_index(wk)
            if n not in window:
                continue
            site_weeks.setdefault(site, {}).setdefault(n, []).append(opp)
        return cls(by_key=dict(opportunities), site_weeks=site_weeks,
                   week_nos=tuple(sorted(window)))

    def opportunities_for(self, site: str, week_no: int) -> list[Opportunity]:
        return self.site_weeks.get(site, {}).get(week_no, [])


@dataclass
class FeatureRow:
    resident: str
    values: dict[str, float]
    evidence: dict[str, int] = field(default_factory=dict)
    missing_flags: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def vector(self, columns: tuple[str, ...] = FEATURE_COLUMNS) -> list[float]:
        return [self.values[c] for c in columns]


@dataclass
class FeatureTable:
    rows: list[FeatureRow]
    columns: tuple[str, ...] = FEATURE_COLUMNS

    def by_resident(self) -> dict[str, FeatureRow]:
        return {r.resident: r for r in self.rows}

    def matrix(self) -> list[list[float]]:
        return [r.vector(self.columns) for r in self.rows]


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def build_features(
    residents: dict[str, Resident],
    events: list[Event],
    view: OpportunityView,
    registry: HouseholdRegistry,
    lookback_weeks: int,
) -> FeatureTable:
    events_by_resident: dict[str, list[Event]] = {}
    for e in events:
        if week_index(e.week) in view.week_nos:
            events_by_resident.setdefault(e.resident, []).append(e)

    household_part = registry.household_participation(events)
    rows: list[FeatureRow] = []

    for rid, resident in residents.items():
        own = events_by_resident.get(rid, [])
        counts = {t: 0 for t in EventType}
        attend_weeks: list[int] = []
        course_counter: dict[str, int] = {}
        for e in own:
            counts[e.event] += 1
            if e.event is EventType.ATTEND:
                attend_weeks.append(e.week_no)
            course_counter[e.course] = course_counter.get(e.course, 0) + 1

        # ---- 机会侧：本人站点在回看窗内的全部/可用机会 ----
        home_rows = [
            opp
            for wk in view.week_nos
            for opp in view.opportunities_for(resident.site, wk)
        ]
        total_opp = len(home_rows)
        usable_opp = sum(1 for o in home_rows if o.usable)
        closed_opp = sum(1 for o in home_rows if o.closed)
        full_opp = sum(1 for o in home_rows if (not o.closed) and (not o.available))
        accessible_weeks = {
            wk
            for wk in view.week_nos
            if any(o.usable for o in view.opportunities_for(resident.site, wk))
        }

        # ---- 跨站替代：本人课程在同周其他站点是否可报 ----
        own_courses = set(course_counter)
        alt_hits = alt_den = 0
        for wk in view.week_nos:
            for course in own_courses:
                alt_den += 1
                if any(
                    o.usable
                    for site in view.site_weeks
                    if site != resident.site
                    for o in view.opportunities_for(site, wk)
                    if o.course == course
                ):
                    alt_hits += 1

        attend_weeks_sorted = sorted(attend_weeks)
        gaps = [
            attend_weeks_sorted[i] - attend_weeks_sorted[i - 1]
            for i in range(1, len(attend_weeks_sorted))
        ]

        half = max(1, len(view.week_nos) // 2)
        early = set(view.week_nos[:half])
        late = set(view.week_nos[-half:])
        early_attend = sum(1 for w in attend_weeks if w in early)
        late_attend = sum(1 for w in attend_weeks if w in late)

        pref_total = sum(course_counter.values())
        concentration = (
            _safe_div(sum(c * c for c in course_counter.values()), pref_total ** 2)
            if pref_total else 0.0
        )

        hh = resident.household
        link = registry.link(hh)
        hh_part = household_part.get(hh, {t.value: 0 for t in EventType})
        if link.state.value == "confirmed" and len(link.residents) > 1:
            hh_total = sum(hh_part.values())
            hh_ratio = _safe_div(
                hh_part.get(EventType.ATTEND.value, 0)
                + hh_part.get(EventType.SIGNUP.value, 0),
                hh_total,
            )
        else:
            hh_ratio = _safe_div(
                counts[EventType.ATTEND] + counts[EventType.SIGNUP],
                sum(counts.values()),
            )

        denom = max(len(accessible_weeks), 1)
        values = {
            "attend_rate": _safe_div(counts[EventType.ATTEND], denom),
            "signup_rate": _safe_div(counts[EventType.SIGNUP], denom),
            "cancel_rate": _safe_div(counts[EventType.CANCEL], denom),
            "no_show_rate": _safe_div(counts[EventType.NO_SHOW], denom),
            "opportunity_coverage": _safe_div(len(accessible_weeks), len(view.week_nos)),
            "closure_exposure": _safe_div(closed_opp, total_opp),
            "fullness_exposure": _safe_div(full_opp, total_opp),
            "mean_attend_gap_weeks": _safe_div(sum(gaps), len(gaps)) if gaps else 0.0,
            "recent_trend": _safe_div(late_attend, len(late)) - _safe_div(early_attend, len(early)),
            "alternative_access": _safe_div(alt_hits, alt_den),
            "preference_concentration": concentration,
            "household_activity_ratio": hh_ratio,
        }

        evidence = {
            "events": len(own),
            "home_opportunities": total_opp,
            "usable_home_opportunities": usable_opp,
            "accessible_weeks": len(accessible_weeks),
            "attend_count": len(attend_weeks_sorted),
        }
        missing: list[str] = []
        notes: list[str] = []
        if total_opp == 0:
            missing.append("home_site_no_opportunity_records")
            notes.append("本站点在回看窗内没有任何机会记录，机会侧证据缺失")
        if not own:
            missing.append("no_events")
            notes.append("回看窗内没有任何参与事件，参与侧证据缺失")
        if not gaps:
            notes.append("出席不足两次，出席间隔不可估，以 0 占位")
        if link.state.value == "pending":
            notes.append("家庭多档案关系待确认，未合并家庭证据")
        rows.append(FeatureRow(rid, values, evidence, tuple(missing), tuple(notes)))

    return FeatureTable(rows=rows)
