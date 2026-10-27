"""Reusable Streamlit UI components for NutriVision AI.

Only HTML/CSS based helpers live here. They are used by ``app.py`` now and by
every ``pages/*.py`` file later, so the look stays consistent across the app.

Note: Streamlit renders each ``st.markdown`` call in its own container, so an
unclosed ``<div>`` cannot wrap other Streamlit widgets. Every component below
therefore emits one complete, self-contained HTML block.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import streamlit as st

from config import STYLE_SHEET

# tone -> css class suffix defined in assets/style.css
_TONES = {"default", "good", "warn", "bad", "accent", "neutral"}


def _clean(html: str) -> str:
    """Strip indentation so Markdown never mistakes HTML for a code block."""
    return "\n".join(line.strip() for line in html.strip().splitlines())


def html_raw(html: str) -> str:
    """Public wrapper around :func:`_clean` for callers building card bodies."""
    return _clean(html)


def _tone(tone: str) -> str:
    return tone if tone in _TONES else "default"


# --------------------------------------------------------------------------- #
# Global styling
# --------------------------------------------------------------------------- #
def load_css(path: Optional[Path] = None) -> None:
    """Inject ``assets/style.css`` into the page (cached across reruns)."""
    css_path = Path(path) if path else STYLE_SHEET
    try:
        css = css_path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - only if the asset is deleted
        st.warning(f"Stylesheet could not be loaded ({exc}). Falling back to default styling.")
        return
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Page furniture
# --------------------------------------------------------------------------- #
def hero(title: str, subtitle: str = "", eyebrow: str = "", icon: str = "") -> None:
    """Large gradient page heading with an optional eyebrow badge and subtitle."""
    eyebrow_html = f'<div class="nv-eyebrow">{eyebrow}</div>' if eyebrow else ""
    subtitle_html = f'<p class="nv-hero-sub">{subtitle}</p>' if subtitle else ""
    icon_html = f'<span class="nv-hero-icon">{icon}</span>' if icon else ""
    st.markdown(
        _clean(
            f"""
            <div class="nv-hero">
                {eyebrow_html}
                <h1 class="nv-hero-title">{icon_html}{title}</h1>
                {subtitle_html}
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def section(title: str, caption: str = "", icon: str = "") -> None:
    """Small section heading used above cards and tables."""
    caption_html = f'<div class="nv-section-caption">{caption}</div>' if caption else ""
    st.markdown(
        _clean(f'<div class="nv-section">{icon} {title}</div>{caption_html}'),
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Cards
# --------------------------------------------------------------------------- #
def metric_card(
    label: str,
    value: str,
    unit: str = "",
    note: str = "",
    tone: str = "default",
) -> None:
    """Compact KPI card (used for BMI, BMR, TDEE, calorie target, ...)."""
    unit_html = f'<span class="nv-metric-unit">{unit}</span>' if unit else ""
    note_html = f'<div class="nv-metric-note">{note}</div>' if note else ""
    st.markdown(
        _clean(
            f"""
            <div class="nv-card nv-metric nv-tone-{_tone(tone)}">
                <div class="nv-metric-label">{label}</div>
                <div class="nv-metric-value">{value}{unit_html}</div>
                {note_html}
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def card(body_html: str, title: str = "", icon: str = "") -> None:
    """Render one complete HTML card.

    ``body_html`` must be closed, valid HTML (for example the output of
    :func:`rows_html` or :func:`stat_rows`-style markup).
    """
    title_html = f'<div class="nv-card-title">{icon} {title}</div>' if title else ""
    st.markdown(
        _clean(f'<div class="nv-card">{title_html}<div class="nv-card-body">{body_html}</div></div>'),
        unsafe_allow_html=True,
    )


def rows_html(rows: Sequence[Tuple[str, str]]) -> str:
    """Return label/value rows as a **complete** HTML block (for :func:`card`)."""
    items = "".join(
        f'<div class="nv-row"><span class="nv-row-label">{label}</span>'
        f'<span class="nv-row-value">{value}</span></div>'
        for label, value in rows
    )
    return _clean(f'<div class="nv-rows">{items}</div>')


def stat_rows(rows: Sequence[Tuple[str, str]]) -> None:
    """Render label/value rows (used for profile details, summaries)."""
    st.markdown(rows_html(rows), unsafe_allow_html=True)


def pill(text: str, tone: str = "neutral") -> str:
    """Return a pill badge as HTML (embed inside :func:`card` bodies)."""
    return f'<span class="nv-pill nv-tone-{_tone(tone)}">{text}</span>'


def notice(text: str, tone: str = "info", icon: str = "ℹ️") -> None:
    """Small inline note (disclaimers, hints, data-source notes)."""
    css_tone = {
        "info": "info",
        "warning": "warn",
        "success": "good",
        "danger": "bad",
    }.get(tone, "info")
    st.markdown(
        _clean(f'<div class="nv-notice nv-tone-{css_tone}">{icon} <span>{text}</span></div>'),
        unsafe_allow_html=True,
    )


def roadmap(items: Iterable[Tuple[str, str, str, str]]) -> None:
    """Render the build roadmap.

    Args:
        items: iterable of ``(label, title, status, description)`` tuples.
    """
    rows = []
    for label, title, status, description in items:
        tone = "good" if "complete" in status.lower() else "neutral"
        rows.append(
            f'<div class="nv-roadmap-item">'
            f'<div class="nv-roadmap-left"><span class="nv-roadmap-label">{label}</span>'
            f'<span class="nv-roadmap-title">{title}</span></div>'
            f'<div class="nv-roadmap-right">{pill(status, tone)}</div></div>'
            f'<div class="nv-roadmap-desc">{description}</div>'
        )
    st.markdown(_clean('<div class="nv-roadmap">' + "".join(rows) + "</div>"), unsafe_allow_html=True)


def error_list(errors: Iterable[str]) -> None:
    """Render validation problems as one readable error block."""
    items = "".join(f"<li>{message}</li>" for message in errors)
    st.markdown(
        _clean(f'<div class="nv-error-box"><strong>Please fix the following:</strong>'
               f"<ul>{items}</ul></div>"),
        unsafe_allow_html=True,
    )


def sidebar_brand(app_name: str, tagline: str, icon: str, phase: str) -> None:
    """Brand block at the top of the sidebar."""
    st.markdown(
        _clean(
            f"""
            <div class="nv-brand">
                <div class="nv-brand-icon">{icon}</div>
                <div>
                    <div class="nv-brand-name">{app_name}</div>
                    <div class="nv-brand-tagline">{tagline}</div>
                </div>
            </div>
            <div class="nv-sidebar-phase">{phase}</div>
            """
        ),
        unsafe_allow_html=True,
    )


def page_footer(phase: str, build: str, note: str) -> None:
    """Footer shown at the bottom of every page (Phase 8 polish).

    One place to state the build, the current phase and the "estimates only"
    reminder, so all seven screens say the same thing.
    """
    st.divider()
    st.caption(f"**NutriVision AI** · build {build} · {phase}")
    st.caption(note)


def key_value_grid(rows: Sequence[Tuple[str, str]], columns: int = 2) -> None:
    """Render ``(label, value)`` pairs as a compact multi-column grid.

    Unlike :func:`stat_rows` this keeps long values (timestamps, backend names)
    on one line and fills the grid left-to-right, which reads better for the
    Profile page's storage summary.
    """
    rows = list(rows)
    for start in range(0, len(rows), columns):
        chunk = rows[start : start + columns]
        columns_ctx = st.columns(columns, gap="medium")
        for ctx, (label, value) in zip(columns_ctx, chunk):
            with ctx:
                st.markdown(
                    _clean(
                        f'<div class="nv-kv"><span class="nv-kv-label">{label}</span>'
                        f'<span class="nv-kv-value">{value}</span></div>'
                    ),
                    unsafe_allow_html=True,
                )
