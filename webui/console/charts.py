"""Small server-side SVG charts; every mark carries a data-tip read by the page's hover layer."""
from __future__ import annotations

import math

from django.utils.html import escape
from django.utils.safestring import mark_safe

INK_MUTED = "#6b7486"
GRID = "#e3e6eb"
STATUS_STYLE = {
    "alert": {"fill": "#e1261d", "shape": "triangle", "label": "Flag kept"},
    "vetoed": {"fill": "#c9821b", "shape": "diamond", "label": "Vetoed by inspector"},
    "clear": {"fill": "#b7c0c9", "shape": "circle", "label": "No flag"},
}


def _axis(value: float, ticks: int = 4) -> tuple[float, float]:
    """An axis maximum and a whole tick step of 1, 2 or 5 times a power of ten."""
    if value <= 0:
        return ticks, 1
    raw = max(value / ticks, 1)
    power = 10 ** math.floor(math.log10(raw))
    step = next(s * power for s in (1, 2, 5, 10) if s * power >= raw)
    return step * math.ceil(value / step), step


def _shape(kind: str, x: float, y: float, r: float, fill: str, tip: str, extra: str = "") -> str:
    """One status marker; a different shape per status so color never carries meaning alone."""
    common = f'fill="{fill}" stroke="#fff" stroke-width="2" data-tip="{escape(tip)}" {extra}'
    if kind == "triangle":
        h = r * 1.25
        return f'<polygon points="{x:.1f},{y - h:.1f} {x + h:.1f},{y + h * .8:.1f} {x - h:.1f},{y + h * .8:.1f}" {common}/>'
    if kind == "diamond":
        return f'<polygon points="{x:.1f},{y - r:.1f} {x + r:.1f},{y:.1f} {x:.1f},{y + r:.1f} {x - r:.1f},{y:.1f}" {common}/>'
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r * .7:.1f}" {common}/>'


def legend_icon(status: str) -> str:
    """A small marker matching the map, for legends and tables."""
    style = STATUS_STYLE[status]
    return mark_safe(f'<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">'
                     f'{_shape(style["shape"], 8, 8.5, 6, style["fill"], "")}</svg>')


def bars(labels: list[str], series: list[tuple[str, str, list[float]]], tips: list[str], height: int = 220,
         width: int = 760, tick_every: int = 1) -> str:
    """Vertical bars, stacked when several series are given, with one hover target per column."""
    left, right, top, bottom = 40, 8, 12, 28
    plot_w, plot_h = width - left - right, height - top - bottom
    totals = [sum(s[2][i] for s in series) for i in range(len(labels))]
    top_value, step = _axis(max(totals, default=0))
    slot = plot_w / max(len(labels), 1)
    bar_w = max(min(slot * 0.62, 28), 2)
    y = lambda v: top + plot_h - v / top_value * plot_h
    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">']
    for i in range(int(top_value / step) + 1):
        value = step * i
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(value):.1f}" y2="{y(value):.1f}" stroke="{GRID}"/>'
                     f'<text x="{left - 6}" y="{y(value) + 4:.1f}" text-anchor="end" class="tick">{value:g}</text>')
    for i, label in enumerate(labels):
        x = left + slot * i + (slot - bar_w) / 2
        base = 0.0
        for _, color, values in series:
            value = values[i]
            if value <= 0:
                continue
            y1, y0 = y(base + value), y(base)
            # 2px surface gap between stacked segments
            gap = 2 if base > 0 else 0
            parts.append(f'<rect x="{x:.1f}" y="{y1:.1f}" width="{bar_w:.1f}" height="{max(y0 - y1 - gap, 1):.1f}" '
                         f'rx="2" fill="{color}"/>')
            base += value
        parts.append(f'<rect x="{left + slot * i:.1f}" y="{top}" width="{slot:.1f}" height="{plot_h}" class="hit" '
                     f'data-tip="{escape(tips[i])}"/>')
        if i % tick_every == 0:
            parts.append(f'<text x="{left + slot * i + slot / 2:.1f}" y="{height - 8}" text-anchor="middle" '
                         f'class="tick">{escape(label)}</text>')
    parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(0):.1f}" y2="{y(0):.1f}" stroke="#9aa3b2"/></svg>')
    return mark_safe("".join(parts))


def line(labels: list[str], values: list[float | None], tips: list[str], color: str = "#41609e", height: int = 220,
         width: int = 760, percent: bool = True, tick_every: int = 1) -> str:
    """One line over a 0-1 axis with markers and one hover target per point."""
    left, right, top, bottom = 40, 8, 12, 28
    plot_w, plot_h = width - left - right, height - top - bottom
    slot = plot_w / max(len(labels), 1)
    x = lambda i: left + slot * i + slot / 2
    y = lambda v: top + plot_h - v * plot_h
    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">']
    for i in range(5):
        value = i / 4
        tick = f"{value:.0%}" if percent else f"{value:g}"
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(value):.1f}" y2="{y(value):.1f}" stroke="{GRID}"/>'
                     f'<text x="{left - 6}" y="{y(value) + 4:.1f}" text-anchor="end" class="tick">{tick}</text>')
    points = [(x(i), y(v)) for i, v in enumerate(values) if v is not None and v == v]
    if points:
        path = " ".join(f"{'M' if i == 0 else 'L'}{px:.1f},{py:.1f}" for i, (px, py) in enumerate(points))
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>')
    for i, v in enumerate(values):
        if v is not None and v == v:
            parts.append(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="4" fill="{color}" stroke="#fff" stroke-width="2"/>')
        parts.append(f'<rect x="{left + slot * i:.1f}" y="{top}" width="{slot:.1f}" height="{plot_h}" class="hit" '
                     f'data-tip="{escape(tips[i])}"/>')
        if i % tick_every == 0:
            parts.append(f'<text x="{x(i):.1f}" y="{height - 8}" text-anchor="middle" class="tick">{escape(labels[i])}</text>')
    parts.append("</svg>")
    return mark_safe("".join(parts))


def meter(value: float | None) -> str:
    """A thin horizontal bar for a probability in a table cell."""
    if value is None or value != value:
        return mark_safe('<span class="muted">—</span>')
    return mark_safe(f'<span class="meter"><span style="width:{value * 100:.0f}%"></span></span>'
                     f'<span class="num">{value:.0%}</span>')
