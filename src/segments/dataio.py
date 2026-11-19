"""CSV/JSON 资料加载与基础校验。

加载器只做结构与类型校验，不做任何业务推断；字段为空时保留为
``None``，由特征工程按"缺失即未知"处理。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .models import (
    ContactPermission,
    Event,
    EventType,
    Opportunity,
    Resident,
    Worker,
    optional,
    required,
)


@dataclass
class Policy:
    reserved_share: float          # 欠覆盖片区保底名额占比
    lookback_weeks: int            # 回看周数
    maximum_repeat_contacts: int   # 同一对象单轮最多外呼次数

    @classmethod
    def load(cls, path: str | Path) -> "Policy":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("policy.json 顶层必须是对象")
        share = float(raw["reserved_share"])
        if not 0.0 <= share <= 1.0:
            raise ValueError("reserved_share 必须落在 [0,1]")
        lookback = int(raw["lookback_weeks"])
        if lookback <= 0:
            raise ValueError("lookback_weeks 必须为正")
        max_repeat = int(raw["maximum_repeat_contacts"])
        if max_repeat < 0:
            raise ValueError("maximum_repeat_contacts 不能为负")
        return cls(share, lookback, max_repeat)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"资料为空: {path}")
    return rows


def load_residents(path: str | Path) -> dict[str, Resident]:
    out: dict[str, Resident] = {}
    for row in _read_csv(path):
        rid = required(row.get("resident", ""), "resident")
        if rid in out:
            raise ValueError(f"居民标识重复: {rid}")
        out[rid] = Resident(
            resident=rid,
            household=required(row.get("household", ""), "household"),
            site=required(row.get("site", ""), "site"),
            group=required(row.get("group", ""), "group"),
            contact_permission=ContactPermission.parse(row.get("contact_permission", "")),
            preferred_time=required(row.get("preferred_time", ""), "preferred_time"),
        )
    return out


def load_opportunities(path: str | Path) -> dict[tuple[str, str, str], Opportunity]:
    out: dict[tuple[str, str, str], Opportunity] = {}
    for row in _read_csv(path):
        key = (
            required(row.get("site", ""), "site"),
            required(row.get("course", ""), "course"),
            required(row.get("week", ""), "week"),
        )
        if key in out:
            raise ValueError(f"机会记录重复: {key}")
        seats = int(required(row.get("seats", ""), "seats"))
        if seats < 0:
            raise ValueError(f"名额不能为负: {key}")
        out[key] = Opportunity(
            site=key[0],
            course=key[1],
            week=key[2],
            available=required(row.get("available", ""), "available") == "1",
            closed=required(row.get("closed", ""), "closed") == "1",
            seats=seats,
        )
    return out


def load_events(path: str | Path, known_residents: set[str]) -> list[Event]:
    out: list[Event] = []
    for row in _read_csv(path):
        rid = required(row.get("resident", ""), "resident")
        if rid not in known_residents:
            # 身份无法校对的事件不能静默并入任何居民，显式报错，
            # 由调用方决定是终止还是隔离。
            raise ValueError(f"事件引用了未知居民标识: {rid}")
        out.append(
            Event(
                resident=rid,
                course=required(row.get("course", ""), "course"),
                week=required(row.get("week", ""), "week"),
                event=EventType.parse(row.get("event", "")),
                reason=optional(row.get("reason", "")),
            )
        )
    return out


def load_staff(path: str | Path) -> dict[str, Worker]:
    out: dict[str, Worker] = {}
    for row in _read_csv(path):
        wid = required(row.get("worker", ""), "worker")
        if wid in out:
            raise ValueError(f"工作人员标识重复: {wid}")
        slots = int(required(row.get("weekly_slots", ""), "weekly_slots"))
        deadline = int(required(row.get("deadline_days", ""), "deadline_days"))
        if slots <= 0 or deadline <= 0:
            raise ValueError(f"工作人员名额与期限必须为正: {wid}")
        out[wid] = Worker(
            worker=wid,
            skill=required(row.get("skill", ""), "skill"),
            site_range=required(row.get("site_range", ""), "site_range"),
            weekly_slots=slots,
            deadline_days=deadline,
        )
    return out


@dataclass
class Dataset:
    residents: dict[str, Resident]
    opportunities: dict[tuple[str, str, str], Opportunity]
    events: list[Event]
    workers: dict[str, Worker]
    policy: Policy

    @property
    def weeks(self) -> list[str]:
        """数据中出现过的全部周标签（按周序号排序）。"""
        labels = {e.week for e in self.events} | {w for (_, _, w) in self.opportunities}
        return sorted(labels, key=lambda w: int(w[1:]))

    @property
    def sites(self) -> set[str]:
        return {r.site for r in self.residents.values()}

    @classmethod
    def load(cls, data_dir: str | Path) -> "Dataset":
        data_dir = Path(data_dir)
        residents = load_residents(data_dir / "residents.csv")
        opportunities = load_opportunities(data_dir / "opportunities.csv")
        events = load_events(data_dir / "events.csv", set(residents))
        workers = load_staff(data_dir / "staff.csv")
        policy = Policy.load(data_dir / "policy.json")
        return cls(residents, opportunities, events, workers, policy)
