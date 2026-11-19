"""稳定性、覆盖与解释检查，以及跨期分群口径比较。

三类检查对应运营的三个担心：
1. **未来周稳定性**：把回看窗前移/后移，居民分群是否稳定；
2. **留出站点稳定性**：每次留出一个站点拟合，该站点居民仍能被
   一致地解释（防止"某站点减少供给就把整站切到负面人群"）；
3. **覆盖与解释**：未分类占比是否可控、每个簇是否都有可报告的
   判别特征。

跨期比较只有在两期分群口径版本一致且簇匹配可靠（Hungarian 式
最优匹配，匹配质量低于门槛）时才输出；否则显式终止对应比较，
由调用方向运营说明"无法可靠匹配历史分群"。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import permutations
from typing import Optional

from .clustering import SegmentationResult, segment
from .features import FeatureTable, OpportunityView, build_features
from .identity import HouseholdRegistry
from .models import Segment


MATCH_RELIABILITY_FLOOR = 0.6  # 低于该匹配度的跨期比较必须终止


@dataclass
class StabilityCheck:
    name: str
    compared_pairs: int
    agreement: float
    coverage: float
    unclassified_rate: float
    explanations_ok: bool
    detail: dict = field(default_factory=dict)


@dataclass
class StabilityReport:
    checks: list[StabilityCheck]

    @property
    def passed(self) -> bool:
        return all(c.agreement >= MATCH_RELIABILITY_FLOOR and c.explanations_ok
                   for c in self.checks)


@dataclass
class RunComparison:
    """两次分群运行的比较结果。"""

    run_a: str
    run_b: str
    version_a: str
    version_b: str
    comparable: bool
    terminated_reason: Optional[str] = None
    cluster_mapping: Optional[dict[int, int]] = None
    match_quality: Optional[float] = None
    behavior_changes: dict[str, tuple[Segment, Segment]] = field(default_factory=dict)
    definition_switch: bool = False

    def changed_residents(self) -> dict[str, tuple[Segment, Segment]]:
        return {r: (a, b) for r, (a, b) in self.behavior_changes.items() if a is not b}


def _agreement(seg_a: dict[str, Segment], seg_b: dict[str, Segment]) -> tuple[int, int, float]:
    common = sorted(set(seg_a) & set(seg_b))
    if not common:
        return 0, 0, 0.0
    agree = sum(1 for r in common if seg_a[r] is seg_b[r])
    return agree, len(common), agree / len(common)


def temporal_stability(
    residents,
    events,
    opportunities,
    registry: HouseholdRegistry,
    lookback_weeks: int,
    *,
    shifts: tuple[int, ...] = (-1, 0, 1),
    pending: Optional[set[str]] = None,
) -> StabilityCheck:
    """在相邻锚点周上重复分群，比较共同居民的分群一致性。"""
    all_weeks = sorted({int(w[1:]) for (_, _, w) in opportunities})
    anchor = max(all_weeks)
    results: list[tuple[int, SegmentationResult]] = []
    for shift in shifts:
        view = OpportunityView.build(opportunities, lookback_weeks, anchor + shift)
        table = build_features(residents, events, view, registry, lookback_weeks)
        results.append((anchor + shift, segment(table, pending_identity_residents=pending)))

    pairs = 0
    agree_sum = 0.0
    unclass_sum = 0
    coverage_sum = 0
    explanations_ok = True
    per_pair: dict[str, float] = {}
    for i in range(len(results) - 1):
        a = results[i][1].segments
        b = results[i + 1][1].segments
        agree, total, ratio = _agreement(a, b)
        if total:
            pairs += 1
            agree_sum += ratio
            per_pair[f"{results[i][0]}->{results[i+1][0]}"] = ratio
        for res in (results[i][1], results[i + 1][1]):
            counts = res.segment_counts()
            unclass_sum += counts[Segment.UNCLASSIFIED] / max(len(res.segments), 1)
            coverage_sum += sum(
                v for k, v in counts.items() if k is not Segment.UNCLASSIFIED
            ) / max(len(res.segments), 1)
            explanations_ok &= all(len(p.discriminators) >= 2 for p in res.profiles)
    n_runs = len(results)
    return StabilityCheck(
        name="temporal_future_weeks",
        compared_pairs=pairs,
        agreement=agree_sum / pairs if pairs else 0.0,
        coverage=coverage_sum / (2 * n_runs) if n_runs else 0.0,
        unclassified_rate=unclass_sum / (2 * n_runs) if n_runs else 0.0,
        explanations_ok=explanations_ok,
        detail={"pair_agreement": per_pair},
    )


def held_out_site_stability(
    residents,
    events,
    opportunities,
    registry: HouseholdRegistry,
    lookback_weeks: int,
    *,
    pending: Optional[set[str]] = None,
) -> StabilityCheck:
    """逐站点留出：只用其余站点拟合特征分布，再给留出站点居民贴标签。

    实现方式：全量拟合一次作为参照；每次把一个站点的居民从聚类
    样本中剔除后重新拟合，再按最近中心把留出居民归入既有簇，
    检查与参照分群是否一致。这样站点的供给变化不会单独决定该站
    居民的分群（中心由其他站点共同决定）。
    """
    view = OpportunityView.build(opportunities, lookback_weeks)
    full_table = build_features(residents, events, view, registry, lookback_weeks)
    reference = segment(full_table, pending_identity_residents=pending)

    sites = sorted({r.site for r in residents.values()})
    agree_sum = 0.0
    runs = 0
    per_site: dict[str, float] = {}
    for site in sites:
        held = {rid for rid, r in residents.items() if r.site == site}
        kept_rows = [row for row in full_table.rows if row.resident not in held]
        if len(kept_rows) < 2:
            continue
        partial = segment(
            FeatureTable(rows=kept_rows, columns=full_table.columns),
            pending_identity_residents=pending,
        )
        # 用 partial 的标准化中心对留出居民做最近中心指派
        means = partial.feature_means
        stds = partial.feature_stds
        centers = {p.cluster_id: p.standardized_centroid for p in partial.profiles}
        if not centers:
            continue
        agree = total = 0
        for row in full_table.rows:
            if row.resident not in held:
                continue
            if reference.segments[row.resident] is Segment.UNCLASSIFIED:
                continue  # 未分类不参与稳定性对错统计
            total += 1
            vec = {
                c: (row.values[c] - means[c]) / stds[c]
                for c in full_table.columns
            }
            best_cid, best_dist = None, math.inf
            for cid, center in centers.items():
                dist = sum((vec[c] - center[c]) ** 2 for c in full_table.columns)
                if dist < best_dist:
                    best_dist, best_cid = dist, cid
            assigned = next(p.segment for p in partial.profiles if p.cluster_id == best_cid)
            if assigned is reference.segments[row.resident]:
                agree += 1
        if total:
            ratio = agree / total
            per_site[site] = ratio
            agree_sum += ratio
            runs += 1

    counts = reference.segment_counts()
    return StabilityCheck(
        name="held_out_sites",
        compared_pairs=runs,
        agreement=agree_sum / runs if runs else 0.0,
        coverage=sum(v for k, v in counts.items() if k is not Segment.UNCLASSIFIED)
        / max(len(reference.segments), 1),
        unclassified_rate=counts[Segment.UNCLASSIFIED] / max(len(reference.segments), 1),
        explanations_ok=all(len(p.discriminators) >= 2 for p in reference.profiles),
        detail={"site_agreement": per_site},
    )


def _best_cluster_mapping(
    result_a: SegmentationResult, result_b: SegmentationResult
) -> tuple[Optional[dict[int, int]], float]:
    """在共同居民上求簇间最优一一匹配，返回映射与匹配质量。"""
    common = sorted(set(result_a.labels) & set(result_b.labels))
    common = [r for r in common if result_a.labels[r] != -1 and result_b.labels[r] != -1]
    if not common:
        return None, 0.0
    ids_a = sorted({result_a.labels[r] for r in common})
    ids_b = sorted({result_b.labels[r] for r in common})
    if len(ids_a) != len(ids_b):
        # 簇数不同，先按列对齐，允许部分匹配；质量会自然下降。
        pass
    overlap: dict[tuple[int, int], int] = {}
    for r in common:
        overlap[(result_a.labels[r], result_b.labels[r])] = (
            overlap.get((result_a.labels[r], result_b.labels[r]), 0) + 1
        )
    best_map, best_hits = None, -1
    for perm in permutations(ids_b, min(len(ids_a), len(ids_b))):
        mapping = dict(zip(ids_a, perm))
        hits = sum(overlap.get((a, b), 0) for a, b in mapping.items())
        if hits > best_hits:
            best_hits, best_map = hits, mapping
    quality = best_hits / len(common)
    return best_map, quality


def compare_runs(
    run_a_id: str,
    result_a: SegmentationResult,
    run_b_id: str,
    result_b: SegmentationResult,
    reliability_floor: float = MATCH_RELIABILITY_FLOOR,
) -> RunComparison:
    """跨期比较入口。不可靠时终止比较而不是硬给变化名单。

    ``reliability_floor`` 允许在更严格的运营口径下提高门槛；默认值
    来自模块常量，业务代码不应为了"得到比较结果"而调低它。
    """
    if result_a.version != result_b.version:
        return RunComparison(
            run_a=run_a_id, run_b=run_b_id,
            version_a=result_a.version, version_b=result_b.version,
            comparable=False, definition_switch=True,
            terminated_reason=(
                f"分群口径版本不一致（{result_a.version} vs {result_b.version}），"
                "属于分群口径切换而非组内行为变化，终止跨期比较"
            ),
        )
    mapping, quality = _best_cluster_mapping(result_a, result_b)
    if mapping is None or quality < reliability_floor:
        return RunComparison(
            run_a=run_a_id, run_b=run_b_id,
            version_a=result_a.version, version_b=result_b.version,
            comparable=False, cluster_mapping=mapping, match_quality=quality,
            terminated_reason=(
                f"历史簇与当前簇匹配质量 {quality:.2f} 低于门槛 "
                f"{reliability_floor:.2f}，无法可靠匹配历史分群，终止跨期比较"
            ),
        )
    common = sorted(set(result_a.segments) & set(result_b.segments))
    changes = {r: (result_a.segments[r], result_b.segments[r]) for r in common}
    return RunComparison(
        run_a=run_a_id, run_b=run_b_id,
        version_a=result_a.version, version_b=result_b.version,
        comparable=True, cluster_mapping=mapping, match_quality=quality,
        behavior_changes=changes,
    )


def run_stability_report(
    residents, events, opportunities, registry, lookback_weeks,
    pending: Optional[set[str]] = None,
) -> StabilityReport:
    return StabilityReport([
        temporal_stability(residents, events, opportunities, registry,
                           lookback_weeks, pending=pending),
        held_out_site_stability(residents, events, opportunities, registry,
                                lookback_weeks, pending=pending),
    ])
