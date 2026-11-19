"""测试公共工具：在临时目录里构造合成资料，不依赖仓库夹具的具体规模。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

HEADERS = {
    "residents": ["resident", "household", "site", "group",
                  "contact_permission", "preferred_time"],
    "opportunities": ["site", "course", "week", "available", "closed", "seats"],
    "events": ["resident", "course", "week", "event", "reason"],
    "staff": ["worker", "skill", "site_range", "weekly_slots", "deadline_days"],
}


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def build_dataset(
    tmp: Path,
    *,
    residents: list[list],
    events: list[list],
    opportunities: list[list],
    staff: list[list],
    policy: dict | None = None,
) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    write_csv(tmp / "residents.csv", HEADERS["residents"], residents)
    write_csv(tmp / "events.csv", HEADERS["events"], events)
    write_csv(tmp / "opportunities.csv", HEADERS["opportunities"], opportunities)
    write_csv(tmp / "staff.csv", HEADERS["staff"], staff)
    (tmp / "policy.json").write_text(
        json.dumps(policy or {"reserved_share": 0.25,
                              "lookback_weeks": 8,
                              "maximum_repeat_contacts": 1}),
        encoding="utf-8",
    )
    return tmp


def four_week_opportunities(sites: list[str], courses=("c0", "c1"),
                            weeks=range(8), closed: set | None = None,
                            full: set | None = None) -> list[list]:
    """每个站点每周每课程一行；closed/full 元素为 (site, course, week)。"""
    closed, full = closed or set(), full or set()
    rows = []
    for site in sites:
        for wk in weeks:
            for course in courses:
                key = (site, course, f"w{wk}")
                is_closed = key in closed
                is_full = key in full
                rows.append([
                    site, course, f"w{wk}",
                    0 if (is_closed or is_full) else 1,
                    1 if is_closed else 0,
                    10,
                ])
    return rows
