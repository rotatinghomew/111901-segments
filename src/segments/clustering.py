"""无先验阈值的可解释分群。

方法选择：层次凝聚聚类（自底向上，Ward 式组内平方和增量），配合
z-score 标准化。它不要求事先写出"出席率低于多少算流失"这类阈值，
分多少组也由数据上的轮廓系数决定，而不是人工拍板。

输出包括：
* 每个居民的簇号与可解释分群名；
* 每个簇的中心特征、相对总体的判别方向（高/低于均值多少个标准差），
  说明分群究竟是被可选机会、出席间隔、活动偏好还是变动趋势拉开的；
* 证据不足的居民归入 ``unclassified``，不编造稳定标签；
* 小簇（低于由数据规模决定的支撑下限）整体回落为 ``unclassified``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .features import FeatureTable
from .models import SEGMENTATION_VERSION, Segment


# 证据不足的判定只依赖"有没有证据"这一事实，不依赖行为阈值。
MIN_EVENTS_FOR_LABEL = 1
MIN_USABLE_OPPORTUNITIES = 1


class UnclassifiedReason(str, Enum):
    NO_EVENTS = "insufficient_evidence_no_events"
    NO_USABLE_OPPORTUNITY = "insufficient_evidence_no_usable_opportunity"
    SMALL_CLUSTER = "cluster_below_support_floor"
    IDENTITY_PENDING = "identity_relationship_pending"


@dataclass
class ClusterProfile:
    cluster_id: int
    size: int
    centroid: dict[str, float]                 # 原始量纲的中心
    standardized_centroid: dict[str, float]   # 相对总体的 z 分数中心
    discriminators: list[tuple[str, float, str]]  # (特征, z值, 方向解释)
    segment: Segment
    naming_rationale: str


@dataclass
class SegmentationResult:
    version: str
    columns: tuple[str, ...]
    k: int
    labels: dict[str, int]                     # resident -> cluster id（未分类为 -1）
    segments: dict[str, Segment]               # resident -> 分群
    profiles: list[ClusterProfile]
    unclassified_reasons: dict[str, UnclassifiedReason]
    feature_means: dict[str, float]
    feature_stds: dict[str, float]
    silhouette: float
    support_floor: int
    # 家庭关系待确认的居民：可按本人证据分群，但家庭维度证据未合并，
    # 标签在身份确认前不作为跨期稳定标签使用。
    pending_identity: frozenset[str] = frozenset()
    notes: list[str] = field(default_factory=list)

    def segment_counts(self) -> dict[Segment, int]:
        out: dict[Segment, int] = {s: 0 for s in Segment}
        for seg in self.segments.values():
            out[seg] += 1
        return out


def _standardize(matrix: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    n = len(matrix)
    d = len(matrix[0])
    means = [sum(row[j] for row in matrix) / n for j in range(d)]
    stds = []
    for j in range(d):
        var = sum((row[j] - means[j]) ** 2 for row in matrix) / n
        std = math.sqrt(var)
        stds.append(std if std > 1e-12 else 1.0)  # 常数列不参与拉开距离
    z = [[(row[j] - means[j]) / stds[j] for j in range(d)] for row in matrix]
    return z, means, stds


def _euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _ward_merge_cost(
    members_a: list[int], members_b: list[int], z: list[list[float]], d: int
) -> float:
    """合并两个簇带来的组内平方和增量（Ward 准则）。"""
    na, nb = len(members_a), len(members_b)
    cost = 0.0
    for j in range(d):
        ca = sum(z[i][j] for i in members_a) / na
        cb = sum(z[i][j] for i in members_b) / nb
        cost += (ca - cb) ** 2
    return cost * na * nb / (na + nb)


def agglomerative(z: list[list[float]], k: int) -> list[int]:
    """自底向上凝聚到 k 个簇，返回每个样本的簇号。"""
    n = len(z)
    clusters: list[list[int]] = [[i] for i in range(n)]
    d = len(z[0])
    while len(clusters) > k:
        best = None
        best_cost = math.inf
        m = len(clusters)
        for a in range(m):
            for b in range(a + 1, m):
                cost = _ward_merge_cost(clusters[a], clusters[b], z, d)
                if cost < best_cost:
                    best_cost = cost
                    best = (a, b)
        a, b = best
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    labels = [0] * n
    for cid, members in enumerate(clusters):
        for i in members:
            labels[i] = cid
    return labels


def silhouette_score(z: list[list[float]], labels: list[int]) -> float:
    """轮廓系数（样本规模小，直接 O(n^2) 计算）。"""
    n = len(z)
    if n <= 1:
        return 0.0
    by_cluster: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        by_cluster.setdefault(lab, []).append(i)
    if len(by_cluster) <= 1:
        return 0.0
    scores = []
    for i in range(n):
        own = by_cluster[labels[i]]
        a_i = (
            sum(_euclidean(z[i], z[j]) for j in own if j != i) / (len(own) - 1)
            if len(own) > 1 else 0.0
        )
        b_i = math.inf
        for lab, members in by_cluster.items():
            if lab == labels[i]:
                continue
            b_i = min(b_i, sum(_euclidean(z[i], z[j]) for j in members) / len(members))
        if math.isinf(b_i):
            continue
        denom = max(a_i, b_i)
        scores.append(0.0 if denom == 0 else (b_i - a_i) / denom)
    return sum(scores) / len(scores) if scores else 0.0


def _support_floor(n: int) -> int:
    """支撑下限随数据规模走：约 5% 且不少于 2，避免把个例当稳定人群。"""
    return max(2, math.ceil(n * 0.05))


def _name_cluster(profile_z: dict[str, float]) -> tuple[Segment, str]:
    """依据簇中心相对总体的判别方向命名，而不是按绝对阈值切人。"""
    closure = profile_z.get("closure_exposure", 0.0)
    fullness = profile_z.get("fullness_exposure", 0.0)
    coverage = profile_z.get("opportunity_coverage", 0.0)
    attend = profile_z.get("attend_rate", 0.0)
    no_show = profile_z.get("no_show_rate", 0.0)
    gap = profile_z.get("mean_attend_gap_weeks", 0.0)
    trend = profile_z.get("recent_trend", 0.0)
    cancel = profile_z.get("cancel_rate", 0.0)

    # 机会受限：供给侧指标（闭馆/满员暴露高、可及覆盖低、取消多）最突出，
    # 而不是出席意愿指标最差。停课不是失去兴趣的证据。
    supply_pressure = max(closure, fullness) - max(no_show, 0.0)
    if (closure > 0.5 or fullness > 0.5 or cancel > 0.5) and supply_pressure > -0.2:
        return (
            Segment.OPPORTUNITY_CONSTRAINED,
            "该组闭馆/满员暴露与取消率显著高于总体，而意愿类指标并非最差，"
            "判定为机会受限而非失去兴趣",
        )
    if attend > 0.25 and no_show <= 0.25:
        return (
            Segment.SUSTAINED,
            "该组出席率显著高于总体、失约率不突出，判定为持续参与",
        )
    if no_show > 0.25 or gap > 0.25 or trend < -0.25:
        return (
            Segment.SHORT_BREAK,
            "该组失约率/出席间隔/走弱趋势高于总体，判定为短期中断，需低频关怀确认原因",
        )
    # 其余形态（例如只报名未出席）也属于短期中断观察对象。
    return (
        Segment.SHORT_BREAK,
        "该组参与信号弱但无供给侧受困证据，按短期中断观察而非流失定性",
    )


def segment(
    table: FeatureTable,
    *,
    max_k: int = 6,
    pending_identity_residents: Optional[set[str]] = None,
) -> SegmentationResult:
    pending_identity_residents = pending_identity_residents or set()
    rows = table.rows
    n = len(rows)
    support_floor = _support_floor(n)

    # 1) 先按证据有无摘出"未分类"，不参与聚类。
    # 注意：家庭关系待确认只阻止跨档案的家庭证据合并（见 features 层），
    # 并不阻止居民按本人证据获得分群；只有身份本身无法校对者才必须未分类。
    # 本站点没有可及机会时，若仍有机会记录（闭馆/满员）或本人事件，
    # 属于供给侧受困而非证据不足，交给聚类识别为机会受限；只有"连记录
    # 都没有且本人也无事件"才是真正的证据不足。
    eligible: list[str] = []
    unclassified_reasons: dict[str, UnclassifiedReason] = {}
    for row in rows:
        has_events = row.evidence.get("events", 0) >= MIN_EVENTS_FOR_LABEL
        has_any_opportunity = row.evidence.get("home_opportunities", 0) > 0
        if not has_events and not has_any_opportunity:
            unclassified_reasons[row.resident] = UnclassifiedReason.NO_EVENTS
        elif (
            row.evidence.get("usable_home_opportunities", 0) < MIN_USABLE_OPPORTUNITIES
            and not has_events
        ):
            unclassified_reasons[
                row.resident
            ] = UnclassifiedReason.NO_USABLE_OPPORTUNITY
        else:
            eligible.append(row.resident)

    if len(eligible) < 2:
        for rid in [r.resident for r in rows]:
            unclassified_reasons.setdefault(rid, UnclassifiedReason.NO_EVENTS)
        return SegmentationResult(
            version=SEGMENTATION_VERSION,
            columns=table.columns, k=0,
            labels={r.resident: -1 for r in rows},
            segments={r.resident: Segment.UNCLASSIFIED for r in rows},
            profiles=[], unclassified_reasons=unclassified_reasons,
            feature_means={c: 0.0 for c in table.columns},
            feature_stds={c: 1.0 for c in table.columns},
            silhouette=0.0, support_floor=support_floor,
            pending_identity=frozenset(pending_identity_residents),
            notes=["可分类样本不足 2，整体保留为未分类"],
        )

    sub_table = FeatureTable(
        rows=[table.by_resident()[rid] for rid in eligible], columns=table.columns
    )
    z, means, stds = _standardize(sub_table.matrix())

    # 2) k 由轮廓系数在数据上选择，上限受样本量约束，不预设业务阈值。
    upper_k = min(max_k, len(eligible))
    best = None
    for k in range(2, upper_k + 1):
        lab = agglomerative(z, k)
        score = silhouette_score(z, lab)
        if best is None or score > best[0]:
            best = (score, k, lab)
    silhouette, k, sub_labels = best

    # 3) 小簇回落为未分类。
    cluster_members: dict[int, list[int]] = {}
    for i, lab in enumerate(sub_labels):
        cluster_members.setdefault(lab, []).append(i)
    labels: dict[str, int] = {r.resident: -1 for r in rows}
    for cid, members in cluster_members.items():
        if len(members) < support_floor:
            for i in members:
                rid = eligible[i]
                unclassified_reasons[rid] = UnclassifiedReason.SMALL_CLUSTER
        else:
            for i in members:
                labels[eligible[i]] = cid

    # 4) 簇画像与命名。
    profiles: list[ClusterProfile] = []
    segments: dict[str, Segment] = {r.resident: Segment.UNCLASSIFIED for r in rows}
    for cid, members in sorted(cluster_members.items()):
        kept = [i for i in members if labels[eligible[i]] != -1]
        if not kept:
            continue
        centroid_raw = {
            col: sum(sub_table.rows[i].values[col] for i in kept) / len(kept)
            for col in table.columns
        }
        centroid_z = {
            col: sum(z[i][j] for i in kept) / len(kept)
            for j, col in enumerate(table.columns)
        }
        ranked = sorted(
            ((col, centroid_z[col]) for col in table.columns),
            key=lambda kv: abs(kv[1]), reverse=True,
        )
        discriminators = [
            (col, val, "高于总体" if val >= 0 else "低于总体")
            for col, val in ranked[:4]
        ]
        seg, rationale = _name_cluster(centroid_z)
        profiles.append(ClusterProfile(
            cluster_id=cid, size=len(kept),
            centroid=centroid_raw, standardized_centroid=centroid_z,
            discriminators=discriminators, segment=seg, naming_rationale=rationale,
        ))
        for i in kept:
            segments[eligible[i]] = seg

    return SegmentationResult(
        version=SEGMENTATION_VERSION,
        columns=table.columns, k=k, labels=labels, segments=segments,
        profiles=profiles, unclassified_reasons=unclassified_reasons,
        feature_means={c: m for c, m in zip(table.columns, means)},
        feature_stds={c: s for c, s in zip(table.columns, stds)},
        silhouette=silhouette, support_floor=support_floor,
        pending_identity=frozenset(pending_identity_residents),
        notes=[f"在 k=2..{upper_k} 中按轮廓系数选择 k={k}，轮廓系数={silhouette:.3f}"],
    )
