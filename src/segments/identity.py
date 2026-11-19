"""按居民标识校对身份，并管理家庭多档案关系。

业务背景：有的家庭用多个档案轮流报名。我们只能在居民**确认**关系
之后，才把同一家庭的参与证据合并解释；待确认（pending）与已撤回
（withdrawn）的关系一律按独立居民处理，联系授权也不共享。

本模块不做标签，只提供：
* 居民标识在事件/机会资料中的校对结果；
* 家庭关系状态机（pending -> confirmed/withdrawn，允许撤回）；
* 授权范围内可联系的同住成员判定；
* 家庭层面的参与证据汇总（仅在 confirmed 时供特征层参考）。
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Event, EventType, HouseholdLink, IdentityState, Resident


@dataclass
class ReconciliationReport:
    resident_count: int
    event_count: int
    unknown_event_residents: tuple[str, ...]
    multi_profile_households: dict[str, tuple[str, ...]]
    pending_links: tuple[str, ...]
    confirmed_links: tuple[str, ...]
    withdrawn_links: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return not self.unknown_event_residents


class HouseholdRegistry:
    def __init__(self, residents: dict[str, Resident]):
        self._residents = residents
        members: dict[str, list[str]] = {}
        for r in residents.values():
            members.setdefault(r.household, []).append(r.resident)
        self._links: dict[str, HouseholdLink] = {
            hh: HouseholdLink(household=hh, residents=tuple(sorted(ids)))
            for hh, ids in members.items()
            if len(ids) > 1
        }
        # 单档案家庭也登记一份，便于统一查询，状态视为 confirmed（无第二人可混淆）。
        for hh, ids in members.items():
            if len(ids) == 1:
                self._links[hh] = HouseholdLink(
                    household=hh, residents=tuple(ids), state=IdentityState.CONFIRMED
                )

    @property
    def households(self) -> dict[str, HouseholdLink]:
        return dict(self._links)

    def link(self, household: str) -> HouseholdLink:
        return self._links[household]

    def state_of(self, household: str) -> IdentityState:
        return self._links[household].state

    def confirm(self, household: str, source: str = "resident") -> None:
        """居民确认家庭关系。"""
        self._links[household].transition(IdentityState.CONFIRMED, source, "居民确认")

    def withdraw(self, household: str, source: str = "resident") -> None:
        """居民撤回关系授权；撤回后不再合并证据、不共享授权。"""
        self._links[household].transition(IdentityState.WITHDRAWN, source, "居民撤回")

    def household_of(self, resident_id: str) -> str:
        return self._residents[resident_id].household

    def authorized_household_contacts(self, resident_id: str) -> tuple[str, ...]:
        """返回允许在同一授权范围内联系的同住居民。

        只有 confirmed 关系才允许；pending/withdrawn 都返回空元组，
        调用方绝不能自动把两个居民的预约拼成一个。
        """
        hh = self.household_of(resident_id)
        link = self._links[hh]
        if link.state is not IdentityState.CONFIRMED:
            return ()
        return tuple(x for x in link.residents if x != resident_id)

    def household_participation(
        self, events: list[Event]
    ) -> dict[str, dict[str, int]]:
        """家庭层面的参与证据汇总。

        仅对 confirmed 的多档案家庭合并计数；其余家庭按单档案口径输出，
        避免在身份待确认时编造"轮流报名"之类的解释。
        """
        per_resident: dict[str, dict[str, int]] = {}
        for e in events:
            bucket = per_resident.setdefault(e.resident, {t.value: 0 for t in EventType})
            bucket[e.event.value] += 1
        out: dict[str, dict[str, int]] = {}
        for hh, link in self._links.items():
            totals = {t.value: 0 for t in EventType}
            if link.state is IdentityState.CONFIRMED:
                for rid in link.residents:
                    for k, v in per_resident.get(rid, {}).items():
                        totals[k] += v
            else:
                # 未确认/已撤回：只登记发起查询者自己的证据，不跨档案合并。
                for rid in link.residents:
                    for k, v in per_resident.get(rid, {}).items():
                        totals[k] += v
                totals = dict(totals)
                totals["_unlinked"] = 1
            out[hh] = totals
        return out


def reconcile(
    residents: dict[str, Resident],
    events: list[Event],
    registry: HouseholdRegistry,
) -> ReconciliationReport:
    known = set(residents)
    unknown = tuple(sorted({e.resident for e in events if e.resident not in known}))
    multi = {
        hh: link.residents
        for hh, link in registry.households.items()
        if len(link.residents) > 1
    }
    pending = tuple(sorted(hh for hh, l in registry.households.items()
                           if l.state is IdentityState.PENDING))
    confirmed = tuple(sorted(hh for hh, l in registry.households.items()
                             if l.state is IdentityState.CONFIRMED))
    withdrawn = tuple(sorted(hh for hh, l in registry.households.items()
                             if l.state is IdentityState.WITHDRAWN))
    return ReconciliationReport(
        resident_count=len(residents),
        event_count=len(events),
        unknown_event_residents=unknown,
        multi_profile_households=multi,
        pending_links=pending,
        confirmed_links=confirmed,
        withdrawn_links=withdrawn,
    )
