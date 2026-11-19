"""领域对象与枚举。

所有时间以数据字段中的周标签（如 ``w0``）表示，周序号由
:func:`week_index` 解析，避免在业务代码里散落字符串切片。
空字符串字段一律解析为 ``None``，表示"该资料没有登记值"，
下游不得把缺失值当作业务事实使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


def week_index(label: str) -> int:
    """把 ``w3`` 之类的周标签解析为可比较的整数。"""
    if not isinstance(label, str) or not label.startswith("w"):
        raise ValueError(f"非法周标签: {label!r}")
    return int(label[1:])


def week_label(index: int) -> str:
    if index < 0:
        raise ValueError(f"周序号不能为负: {index}")
    return f"w{index}"


def optional(value: str) -> Optional[str]:
    value = (value or "").strip()
    return value or None


def required(value: str, field_name: str) -> str:
    value = (value or "").strip()
    if not value:
        raise ValueError(f"必填字段缺失: {field_name}")
    return value


class EventType(str, Enum):
    SIGNUP = "signup"
    CANCEL = "cancel"
    ATTEND = "attend"
    NO_SHOW = "no-show"

    @classmethod
    def parse(cls, raw: str) -> "EventType":
        try:
            return cls(required(raw, "event"))
        except ValueError as exc:
            raise ValueError(f"未知事件类型: {raw!r}") from exc


class ContactPermission(str, Enum):
    YES = "yes"          # 允许按登记方式联系
    LIMITED = "limited"  # 受限联系（限授权渠道/频次，仍可分配但需标注）
    NO = "no"            # 不允许主动外呼

    @classmethod
    def parse(cls, raw: str) -> "ContactPermission":
        try:
            return cls(required(raw, "contact_permission"))
        except ValueError as exc:
            raise ValueError(f"未知联系授权: {raw!r}") from exc


class Segment(str, Enum):
    """可解释的四类行为组（名称只描述行为，不做价值判断）。"""

    SUSTAINED = "sustained_participation"        # 持续参与
    OPPORTUNITY_CONSTRAINED = "opportunity_limited"  # 机会受限
    SHORT_BREAK = "short_break"                  # 短期中断
    UNCLASSIFIED = "unclassified"                # 证据不足/小样本/身份待确认


# 分群口径版本：口径本身变化时必须显式升版，便于区分"组内行为变化"
# 与"分群口径切换"。
SEGMENTATION_VERSION = "v1.0"

# 各组对应的支持类型（停课不是居民失去兴趣的证据，因此机会受限组
# 得到的是供给侧支持而非挽回外呼）。
SEGMENT_SUPPORT = {
    Segment.SUSTAINED: "保持供给与偏好匹配，无需主动外呼",
    Segment.OPPORTUNITY_CONSTRAINED: "供给侧调整：恢复/新增站点课程、跨站替代名额",
    Segment.SHORT_BREAK: "低频关怀回访，确认中断原因与可参加时段",
    Segment.UNCLASSIFIED: "先补齐证据/确认身份，不分配稳定标签",
}


@dataclass(frozen=True)
class Resident:
    resident: str
    household: str
    site: str
    group: str  # adult / senior / youth
    contact_permission: ContactPermission
    preferred_time: str

    @property
    def contactable(self) -> bool:
        return self.contact_permission is not ContactPermission.NO


@dataclass(frozen=True)
class Opportunity:
    site: str
    course: str
    week: str
    available: bool
    closed: bool
    seats: int

    @property
    def week_no(self) -> int:
        return week_index(self.week)

    @property
    def usable(self) -> bool:
        """该周该站点该课程是否实际上可报名。闭馆或满员都不可用。"""
        return self.available and not self.closed


@dataclass(frozen=True)
class Event:
    resident: str
    course: str
    week: str
    event: EventType
    reason: Optional[str]

    @property
    def week_no(self) -> int:
        return week_index(self.week)


@dataclass(frozen=True)
class Worker:
    worker: str
    skill: str           # youth / senior / general
    site_range: str      # 常跑范围（单站点标识）
    weekly_slots: int
    deadline_days: int


class IdentityState(str, Enum):
    PENDING = "pending"     # 身份关系尚待居民确认
    CONFIRMED = "confirmed"
    WITHDRAWN = "withdrawn"


@dataclass
class HouseholdLink:
    """同一家庭内两个居民档案之间的关系状态。

    ``confirmed``/``withdrawn`` 只能由居民侧确认或撤回动作驱动；
    未确认（pending）时禁止据此合并预约或共享联系授权。
    """

    household: str
    residents: tuple[str, ...]
    state: IdentityState = IdentityState.PENDING
    history: list[dict] = field(default_factory=list)

    def transition(self, new_state: IdentityState, source: str, note: str = "") -> None:
        if new_state is self.state:
            return
        self.history.append(
            {"from": self.state.value, "to": new_state.value, "source": source, "note": note}
        )
        self.state = new_state


class ActionState(str, Enum):
    RECOMMENDED = "recommended"      # 模型建议，尚未联系
    ACCEPTED = "accepted"            # 居民已接受服务（锁定，不因换标签改约）
    PENDING = "pending"              # 已排期待处理
    COMPLETED = "completed"          # 已执行（保留当时依据）
    SKIPPED = "skipped"              # 本轮不执行（需记录原因）


@dataclass
class Action:
    """一次随访建议/处置。同一对象同一轮最多一条外呼建议。"""

    action_id: str
    resident: str
    worker: Optional[str]
    segment: Segment
    reason_code: str
    state: ActionState
    created_run: str
    basis: dict
    history: list[dict] = field(default_factory=list)

    def transition(self, new_state: ActionState, source: str, note: str = "") -> None:
        self.history.append(
            {"from": self.state.value, "to": new_state.value, "source": source, "note": note}
        )
        self.state = new_state
