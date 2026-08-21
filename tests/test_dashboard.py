"""
Dashboard tests.

A dashboard fails in ways a unit test can catch and a screenshot cannot: a
colour defined only inside a media query so the page renders one theme's text on
the other theme's ground, an external asset the artifact CSP will block, a
number transcribed rather than derived. Those are what is asserted here.

The rendering itself is not tested — no assertion can tell you whether a layout
reads well — but the page's *contract* can be: self-contained, theme-complete,
and showing figures that came out of the run rather than out of the template.
"""

from __future__ import annotations

import json
import re

import pytest

from prismprice.observability.dashboard import build_dashboard, render_dashboard


def make_run(n: int = 6) -> dict:
    return {
        "as_of": "2011-12-09T00:00:00+00:00",
        "n_panel_rows": 1234,
        "may_publish": False,
        "notes": ["unit_cost derived from an assumed 45% gross margin"],
        "unobserved_confounders": ["marketing_spend"],
        "breakers": [
            {
                "name": "one_sided_movement",
                "status": "HALT",
                "halted": True,
                "observed": 0.33,
                "limit": 0.30,
                "detail": "33.3% of the catalogue went up",
            },
            {
                "name": "no_guardrail_violations",
                "status": "PASS",
                "halted": False,
                "observed": 0.0,
                "limit": 0.0,
                "detail": "all prices satisfy the margin floor",
            },
        ],
        "decisions": [
            {
                "sku": f"SKU-{i:03d}",
                "current_price": 10.0 + i,
                "recommended_price": 10.0 + i + (0.5 if i % 2 else -0.5),
                "unit_cost": 5.0,
                "elasticity": -1.5 - i * 0.1,
                "elasticity_confidence": "high" if i % 2 else "low",
                "degradation_rung": 1 if i % 3 else 4,
                "degradation_reason": "OK_OPTIMAL",
                "binding_constraints": [] if i % 3 else ["PP-G005"],
                "expected_units": 100.0,
                "expected_profit": 500.0,
                "price_change_pct": 0.05 if i % 2 else -0.05,
            }
            for i in range(n)
        ],
        "elasticities": {
            f"SKU-{i:03d}": {
                "point": -1.5 - i * 0.1,
                "ci_low": -2.0 - i * 0.1,
                "ci_high": -1.0 - i * 0.1,
                "confidence": "high" if i % 2 else "low",
            }
            for i in range(n)
        },
    }


@pytest.fixture(scope="module")
def html() -> str:
    return build_dashboard(make_run())


# ---------------------------------------------------------------------------
# The page must survive the artifact sandbox
# ---------------------------------------------------------------------------


def test_page_is_self_contained_apart_from_google_fonts(html):
    """A strict CSP blocks every external host except Google Fonts, so anything
    else fails silently and the page renders without it."""
    hosts = set(re.findall(r'https?://([^/"\')\s]+)', html))
    allowed = {"fonts.googleapis.com", "fonts.gstatic.com"}
    assert hosts <= allowed, f"page references blocked hosts: {sorted(hosts - allowed)}"


def test_no_script_or_style_is_loaded_from_a_file(html):
    assert "<script src=" not in html
    assert 'rel="stylesheet" href="https://fonts.googleapis.com' in html
    assert html.count('rel="stylesheet"') == 1


def test_every_font_has_a_real_fallback_stack(html):
    """A silent font fallback to Times is the most common invisible defect."""
    for declaration in re.findall(r"font-family:([^;}]+)", html):
        assert "," in declaration, f"no fallback in font-family:{declaration}"


# ---------------------------------------------------------------------------
# Theme completeness — the classic unreadable-artifact bug
# ---------------------------------------------------------------------------


def test_light_palette_is_defined_on_bare_root(html):
    """The default 'system' setting stamps nothing on the root element, so a
    palette that only exists inside a [data-theme] block never applies."""
    block = html.split(":root{", 1)[1].split("}", 1)[0]
    for token in ("--ground", "--surface", "--ink", "--accent", "--ok", "--stop"):
        assert token in block, f"{token} missing from the bare :root palette"


def test_dark_media_query_is_guarded_against_an_explicit_light_choice(html):
    """Otherwise a dark OS beats a viewer who explicitly chose light."""
    assert ':root:not([data-theme="light"])' in html


def test_dark_is_defined_for_both_the_media_query_and_the_stamp(html):
    """Three viewer states, not two: stamped dark, stamped light, and unstamped
    system — the toggle has to win in both directions."""
    assert "@media (prefers-color-scheme: dark)" in html
    assert ':root[data-theme="dark"]' in html


def test_body_paints_its_own_background(html):
    """The viewer composites the artifact over a ground painted in *its* theme,
    so a transparent body silently borrows the host's."""
    body_rule = html.split("body{", 1)[1].split("}", 1)[0]
    assert "background:var(--" in body_rule


def test_no_colour_is_declared_only_inside_a_theme_block(html):
    """Components style through tokens; a literal inside a theme block would
    apply in one state and vanish in the others."""
    for block in re.findall(r"\[data-theme=\"dark\"\]\{([^}]*)\}", html):
        for declaration in block.split(";"):
            if declaration.strip().startswith("--"):
                continue
            assert "color" not in declaration, f"non-token colour in a theme block: {declaration}"


# ---------------------------------------------------------------------------
# The numbers come from the run
# ---------------------------------------------------------------------------


def _payload(html: str) -> dict:
    raw = html.split("const DATA = ", 1)[1].split(";\n", 1)[0]
    return json.loads(raw)


def test_the_run_is_embedded_rather_than_transcribed(html):
    """A dashboard typed by hand starts agreeing with the system and stops."""
    data = _payload(html)
    assert len(data["decisions"]) == 6
    assert data["summary"]["n_priced"] == 6


def test_rung_counts_are_derived_from_the_decisions(html):
    counts = _payload(html)["summary"]["rung_counts"]
    # i % 3 == 0 gives rung 4: indices 0 and 3 of six.
    assert counts["4"] == 2
    assert counts["1"] == 4


def test_the_halt_verdict_survives_into_the_page(html):
    assert _payload(html)["summary"]["may_publish"] is False
    assert "Publication halted" in html or "may_publish" in html


def test_breakers_are_carried_with_their_reasons(html):
    breakers = _payload(html)["summary"]["breakers"]
    assert len(breakers) == 2
    assert any(b["halted"] for b in breakers)
    assert all(b["detail"] for b in breakers), "a halt with no reason is not actionable"


def test_the_confusion_matrix_is_scored_not_invented(html):
    """It comes from labelled batches, so it must be a real 2x2 that adds up."""
    matrix = _payload(html)["summary"]["classification"]
    grid = matrix["grid"]
    total = sum(sum(row) for row in grid)
    assert total > 0
    assert (
        matrix["true_positive"]
        + matrix["false_positive"]
        + matrix["true_negative"]
        + matrix["false_negative"]
        == total
    )


def test_caveats_travel_with_the_numbers(html):
    """The assumed margin and the unobserved confounders are on the page, not
    only in a commit message."""
    summary = _payload(html)["summary"]
    assert summary["notes"]
    assert "marketing_spend" in summary["unobserved"]


def test_an_empty_run_still_renders(html):
    """A run that priced nothing is a state an operator needs to see, not a
    crash."""
    empty = make_run(0)
    page = build_dashboard(empty)
    assert _payload(page)["summary"]["n_priced"] == 0


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_render_writes_the_page_beside_the_run(tmp_path):
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps(make_run()), encoding="utf-8")
    out = render_dashboard(run_path, tmp_path / "nested" / "dash.html")
    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("<title>")


def test_a_missing_run_says_how_to_produce_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="run_pipeline"):
        render_dashboard(tmp_path / "absent.json", tmp_path / "out.html")


def test_the_title_is_a_name_not_a_caption(html):
    title = html.split("<title>", 1)[1].split("</title>", 1)[0]
    assert 0 < len(title) < 40
    assert "—" not in title and ":" not in title, "a title with an explainer is filler"
