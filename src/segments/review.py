"""周会复盘：区分运营动作、重新参与与机会变化，禁止错误归因。

关键纪律：回访之后发生的出席只能报告为"时间上先后发生"，
不能直接归因于分群模型或某次外呼。所有指标同时给出分子/分母，
并单独列出机会侧变化（站点开课、闭馆、满员），让运营自己判断
参与变化有多少来自供给调整。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import Action, ActionState, Event, EventType, Opportunity, week_index


@dataclass
class ActionSummary:
    recommended: int = 0
    accepted: int = 0
    pending: int = 0
    completed: int = 0
    skipped: int = 0


@dataclass
class OpportunityChange:
    site: str
    usable_before: int
    usable_after: int
    closures_before: int
    closures_after: int
    full_before: int
    full_after: int

    @property
    def net_usable_change(self) -> int:
        return self.usable_after - self.usable_before


@dataclass
class WeeklyReview:
    window_before: tuple[int, int]
    window_after: tuple[int, int]
    action_summary: ActionSummary
    contacted_residents: tuple[str, ...]
    reengaged: tuple[str, ...]
    reengagement_rate_contacted: float
    spontaneous_returns: tuple[str, ...]
    opportunity_changes: list[OpportunityChange]
    attribution_notice: str = field(default="")
    segments_at_run: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "window_before": list(self.window_before),
            "window_after": list(self.window_after),
            "actions": self.action_summary.__dict__,
            "contacted": list(self.contacted_residents),
            "reengaged": list(self.reengaged),
            "reengagement_rate_contacted": round(self.reengagement_rate_contacted, 4),
            "spontaneous_returns": list(self.spontaneous_returns),
            "opportunity_changes": [
                {
                    "site": c.site,
                    "net_usable_change": c.net_usable_change,
                    "usable_before": c.usable_before,
                    "usable_after": c.usable_after,
                    "closures_before": c.closures_before,
                    "closures_after": c.closures_after,
                    "full_before": c.full_before,
                    "full_after": c.full_after,
                }
                for c in self.opportunity_changes
            ],
            "attribution_notice": self.attribution_notice,
        }


def _opp_slice(opportunities, site: str, lo: int, hi: int) -> list[Opportunity]:
    return [
        o for (s, _, w), o in opportunities.items()
        if s == site and lo <= week_index(w) <= hi
    ]


def build_weekly_review(
    *,
    actions: dict[str, Action],
    events: list[Event],
    opportunities: dict[tuple[str, str, str], Opportunity],
    before_window: tuple[int, int],
    after_window: tuple[int, int],
    segments_at_run: dict[str, str],
) -> WeeklyReview:
    summary = ActionSummary()
    contacted: set[str] = set()
    for action in actions.values():
        summary.__dict__[action.state.value] += 1
        # "已联系"只统计实际进入处置链路的动作；recommended 是尚未联系的
        # 模型建议，不能计入已联系，也不能把之后的出席算成回访成果。
        if action.state in (ActionState.ACCEPTED, ActionState.PENDING,
                            ActionState.COMPLETED):
            contacted.add(action.resident)

    attended_after = {
        e.resident for e in events
        if e.event is EventType.ATTEND
        and after_window[0] <= e.week_no <= after_window[1]
    }
    attended_before = {
        e.resident for e in events
        if e.event is EventType.ATTEND
        and before_window[0] <= e.week_no <= before_window[1]
    }
    new_attendees = attended_after - attended_before
    reengaged = tuple(sorted(new_attendees & contacted))
    spontaneous = tuple(sorted(new_attendees - contacted))
    rate = len(reengaged) / len(contacted) if contacted else 0.0

    sites = sorted({s for (s, _, _) in opportunities})
    changes: list[OpportunityChange] = []
    for site in sites:
        before = _opp_slice(opportunities, site, *before_window)
        after = _opp_slice(opportunities, site, *after_window)
        changes.append(OpportunityChange(
            site=site,
            usable_before=sum(1 for o in before if o.usable),
            usable_after=sum(1 for o in after if o.usable),
            closures_before=sum(1 for o in before if o.closed),
            closures_after=sum(1 for o in after if o.closed),
            full_before=sum(1 for o in before if not o.closed and not o.available),
            full_after=sum(1 for o in after if not o.closed and not o.available),
        ))

    notice = (
        "重新参与率仅描述'被联系居民在随后窗口内出现出席'的时间先后关系，"
        "不能归因于分群模型或外呼动作；请结合机会变化与运营动作记录共同判读。"
    )
    return WeeklyReview(
        window_before=before_window, window_after=after_window,
        action_summary=summary,
        contacted_residents=tuple(sorted(contacted)),
        reengaged=reengaged,
        reengagement_rate_contacted=rate,
        spontaneous_returns=spontaneous,
        opportunity_changes=changes,
        attribution_notice=notice,
        segments_at_run=dict(segments_at_run),
    )
