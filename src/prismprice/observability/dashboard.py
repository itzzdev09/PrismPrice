"""
Dashboard generation (L7).

Renders a pipeline run as a self-contained interactive page. The point of
generating it from a module rather than hand-writing HTML is that the numbers on
the page and the numbers in the run report are the same objects — a dashboard
that is transcribed by hand starts agreeing with the system and stops.

What the page leads with
------------------------

The publish verdict, not the profit. This system's whole claim is the difference
between "we returned a price" and "we returned a *good* price", so the first
thing on screen is whether the batch may publish and which circuit breakers had
an opinion. A dashboard that opened with contribution would be answering a
question the operator cannot act on before the more urgent one.

The degradation histogram sits directly beneath it for the same reason
(metrics.md §5): a system that always answers hides its own degradation inside a
healthy-looking success rate.

Cross-filtering
---------------

Clicking a rung bar or a confidence chip filters the decisions table *and*
recomputes the summary tiles. That is the behaviour that makes a dashboard worth
opening rather than reading as a report: the useful question is almost never
"what is the median elasticity" but "what does the median look like among the
SKUs that degraded", and that is one click rather than a new query.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["build_dashboard", "render_dashboard"]


def _summarise(run: dict[str, Any]) -> dict[str, Any]:
    """Derive everything the page displays, so the template holds no arithmetic."""
    decisions = run.get("decisions", [])
    elasticities = run.get("elasticities", {})

    changes = [d["price_change_pct"] for d in decisions if d.get("price_change_pct") is not None]
    rungs = [int(d.get("degradation_rung", 1)) for d in decisions]
    rung_counts = {str(r): sum(1 for x in rungs if x == r) for r in range(1, 6)}

    points = [
        e["point"]
        for e in elasticities.values()
        if e.get("point") is not None and np.isfinite(e["point"])
    ]
    widths = [
        e["ci_high"] - e["ci_low"]
        for e in elasticities.values()
        if e.get("ci_high") is not None and np.isfinite(e.get("ci_high", np.nan))
    ]
    usable = sum(1 for e in elasticities.values() if e.get("confidence") == "high")

    from prismprice.observability.evaluation import elasticity_health

    health = elasticity_health(points, widths, n_declined=len(elasticities) - len(points))

    # The circuit breakers are the only genuine binary classifier here, and one
    # run is one observation — not enough to score anything. So the matrix comes
    # from a labelled simulation where batch status is known by construction.
    #
    # An earlier version paired "did a guardrail bind" with "did the decision
    # degrade" off the real decisions and called that a classifier. It is not:
    # on a rung-4 fallback nothing binds *because the held price passes*, so the
    # pairing scored a working system as a wall of false positives. Two boolean
    # fields are not automatically a prediction and a truth.
    from prismprice.observability.evaluation import score_breakers_on_labelled_batches

    classification = score_breakers_on_labelled_batches().as_dict()

    return {
        "n_priced": len(decisions),
        "n_panel_rows": run.get("n_panel_rows", 0),
        "may_publish": run.get("may_publish", True),
        "breakers": run.get("breakers", []),
        "rung_counts": rung_counts,
        "degraded_share": (sum(1 for r in rungs if r >= 3) / len(rungs)) if rungs else 0.0,
        "median_change": float(np.median(changes)) if changes else 0.0,
        "share_up": float(np.mean([c > 1e-9 for c in changes])) if changes else 0.0,
        "share_down": float(np.mean([c < -1e-9 for c in changes])) if changes else 0.0,
        "elasticity": {
            "median": float(np.median(points)) if points else float("nan"),
            "p10": float(np.quantile(points, 0.10)) if points else float("nan"),
            "p90": float(np.quantile(points, 0.90)) if points else float("nan"),
            "n": len(points),
            "usable": usable,
            "share_usable": usable / max(len(elasticities), 1),
            "health": health,
        },
        "classification": classification,
        "notes": run.get("notes", []),
        "unobserved": run.get("unobserved_confounders", []),
        "as_of": run.get("as_of", ""),
    }


def build_dashboard(run: dict[str, Any]) -> str:
    """Return the full HTML for one pipeline run."""
    summary = _summarise(run)
    payload = json.dumps(
        {
            "summary": summary,
            "decisions": run.get("decisions", []),
            "elasticities": run.get("elasticities", {}),
        },
        default=str,
    )
    return _TEMPLATE.replace("__PAYLOAD__", payload)


def render_dashboard(
    run_path: str | Path = "data/runs/real_run.json",
    output_path: str | Path = "data/runs/dashboard.html",
) -> Path:
    """Read a saved run and write the dashboard beside it."""
    source = Path(run_path)
    if not source.exists():
        raise FileNotFoundError(
            f"{source} not found. Run prismprice.pipeline.run_pipeline() and save its "
            f"as_dict() output there first."
        )
    run = json.loads(source.read_text(encoding="utf-8"))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_dashboard(run), encoding="utf-8")
    return destination


_TEMPLATE = r"""<title>PrismPrice Run Review</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
  :root{
    --ground:#F5F6F8; --surface:#FFFFFF; --surface-2:#EEF1F3;
    --ink:#14171C; --ink-2:#4A545F; --ink-3:#77848F;
    --rule:rgba(15,110,106,.16); --rule-strong:rgba(15,110,106,.30);
    --accent:#0F6E6A; --accent-soft:rgba(15,110,106,.10);
    --ok:#146B3A; --warn:#A85B00; --stop:#A81F1F;
    --ok-soft:rgba(20,107,58,.12); --warn-soft:rgba(168,91,0,.12); --stop-soft:rgba(168,31,31,.12);
    --shadow:0 1px 2px rgba(20,23,28,.05),0 8px 24px rgba(20,23,28,.06);
    --step--1:.78rem; --step-0:.95rem; --step-1:1.15rem; --step-2:1.6rem; --step-3:2.3rem; --step-4:3.1rem;
  }
  @media (prefers-color-scheme: dark){
    :root:not([data-theme="light"]){
      --ground:#0E1116; --surface:#171B21; --surface-2:#1E242B;
      --ink:#E8ECEF; --ink-2:#A3AFB9; --ink-3:#77848F;
      --rule:rgba(93,201,193,.18); --rule-strong:rgba(93,201,193,.32);
      --accent:#5DC9C1; --accent-soft:rgba(93,201,193,.12);
      --ok:#4BBE7B; --warn:#E0913A; --stop:#E56B6B;
      --ok-soft:rgba(75,190,123,.14); --warn-soft:rgba(224,145,58,.14); --stop-soft:rgba(229,107,107,.14);
      --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.35);
    }
  }
  :root[data-theme="dark"]{
    --ground:#0E1116; --surface:#171B21; --surface-2:#1E242B;
    --ink:#E8ECEF; --ink-2:#A3AFB9; --ink-3:#77848F;
    --rule:rgba(93,201,193,.18); --rule-strong:rgba(93,201,193,.32);
    --accent:#5DC9C1; --accent-soft:rgba(93,201,193,.12);
    --ok:#4BBE7B; --warn:#E0913A; --stop:#E56B6B;
    --ok-soft:rgba(75,190,123,.14); --warn-soft:rgba(224,145,58,.14); --stop-soft:rgba(229,107,107,.14);
    --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.35);
  }

  *{box-sizing:border-box}
  body{
    margin:0; background:var(--ground); color:var(--ink);
    font-family:"IBM Plex Sans",ui-sans-serif,system-ui,sans-serif;
    font-size:var(--step-0); line-height:1.55; -webkit-font-smoothing:antialiased;
  }
  .num{font-family:"IBM Plex Mono",ui-monospace,monospace; font-variant-numeric:tabular-nums}
  .wrap{max-width:1240px; margin:0 auto; padding:0 24px 96px}

  /* ---- verdict bar ---- */
  .verdict{position:sticky; top:0; z-index:50; background:var(--surface); border-bottom:1px solid var(--rule-strong)}
  .verdict-in{max-width:1240px; margin:0 auto; padding:18px 24px; display:flex; align-items:center; gap:20px; flex-wrap:wrap}
  .stamp{display:inline-flex; align-items:center; gap:10px; padding:8px 16px; border-radius:2px;
    font-family:"IBM Plex Mono",monospace; font-weight:600; font-size:var(--step--1); letter-spacing:.10em; text-transform:uppercase}
  .stamp.stop{background:var(--stop-soft); color:var(--stop); box-shadow:inset 0 0 0 1px currentColor}
  .stamp.go{background:var(--ok-soft); color:var(--ok); box-shadow:inset 0 0 0 1px currentColor}
  .dot{width:8px;height:8px;border-radius:50%;background:currentColor}
  .verdict h1{font-family:"Newsreader",Georgia,serif; font-weight:600; font-size:var(--step-1); margin:0; letter-spacing:-.01em}
  .verdict .meta{color:var(--ink-3); font-size:var(--step--1); margin-left:auto; display:flex; gap:18px; align-items:center}
  .toggle{background:none;border:1px solid var(--rule-strong);color:var(--ink-2);border-radius:2px;
    padding:6px 12px;font:inherit;font-size:var(--step--1);cursor:pointer}
  .toggle:hover{border-color:var(--accent); color:var(--accent)}
  .toggle:focus-visible{outline:2px solid var(--accent); outline-offset:2px}

  /* ---- headings ---- */
  h2{font-family:"Newsreader",Georgia,serif; font-weight:600; font-size:var(--step-2);
     margin:56px 0 6px; letter-spacing:-.015em; text-wrap:balance}
  h2:first-of-type{margin-top:40px}
  .sub{color:var(--ink-2); max-width:68ch; margin:0 0 22px; font-size:var(--step-0)}
  .eyebrow{font-family:"IBM Plex Mono",monospace; font-size:.68rem; letter-spacing:.16em;
    text-transform:uppercase; color:var(--accent); margin:0 0 4px}

  /* ---- cards & grid ---- */
  .grid{display:grid; gap:16px}
  .g4{grid-template-columns:repeat(auto-fit,minmax(210px,1fr))}
  .g2{grid-template-columns:repeat(auto-fit,minmax(340px,1fr))}
  .card{background:var(--surface); border:1px solid var(--rule); border-radius:3px;
    padding:20px; box-shadow:var(--shadow); position:relative; overflow:hidden}
  .card.stripe::before{content:""; position:absolute; inset:0 auto 0 0; width:3px; background:var(--accent)}
  .card.stripe.is-ok::before{background:var(--ok)}
  .card.stripe.is-warn::before{background:var(--warn)}
  .card.stripe.is-stop::before{background:var(--stop)}
  .label{font-size:var(--step--1); color:var(--ink-2); display:block; margin-bottom:8px}
  .figure{font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums;
    font-size:var(--step-3); font-weight:500; line-height:1.05; letter-spacing:-.02em}
  .foot{font-size:var(--step--1); color:var(--ink-3); margin-top:8px}

  /* ---- bars ---- */
  .bars{display:flex; flex-direction:column; gap:10px; margin-top:4px}
  .bar-row{display:grid; grid-template-columns:132px 1fr 68px; align-items:center; gap:12px;
    background:none;border:0;padding:6px 4px;font:inherit;color:inherit;cursor:pointer;text-align:left;border-radius:2px}
  .bar-row:hover{background:var(--accent-soft)}
  .bar-row:focus-visible{outline:2px solid var(--accent); outline-offset:1px}
  .bar-row[aria-pressed="true"]{background:var(--accent-soft); box-shadow:inset 0 0 0 1px var(--rule-strong)}
  .bar-track{height:22px; background:var(--surface-2); border-radius:2px; overflow:hidden}
  .bar-fill{height:100%; background:var(--accent); transition:width .35s cubic-bezier(.2,.7,.3,1)}
  .bar-fill.is-warn{background:var(--warn)} .bar-fill.is-stop{background:var(--stop)}
  .bar-row .k{font-size:var(--step--1); color:var(--ink-2)}
  .bar-row .v{font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums; text-align:right; font-size:var(--step--1)}

  /* ---- breakers ---- */
  .breaker{display:flex; gap:14px; align-items:flex-start; padding:14px 0; border-top:1px solid var(--rule)}
  .breaker:first-child{border-top:0}
  .breaker .name{font-family:"IBM Plex Mono",monospace; font-size:var(--step--1); min-width:170px}
  .pill{font-family:"IBM Plex Mono",monospace; font-size:.68rem; letter-spacing:.08em; padding:3px 9px;
    border-radius:2px; text-transform:uppercase; font-weight:600; white-space:nowrap}
  .pill.go{background:var(--ok-soft); color:var(--ok)} .pill.stop{background:var(--stop-soft); color:var(--stop)}
  .pill.warn{background:var(--warn-soft); color:var(--warn)} .pill.mute{background:var(--surface-2); color:var(--ink-3)}
  .breaker .why{color:var(--ink-2); font-size:var(--step--1)}

  /* ---- confusion matrix ---- */
  .cm{display:grid; grid-template-columns:auto repeat(2,1fr); gap:4px; margin-top:12px; max-width:380px}
  .cm div{padding:14px 10px; text-align:center; border-radius:2px; font-family:"IBM Plex Mono",monospace;
    font-variant-numeric:tabular-nums}
  .cm .hd{background:none; color:var(--ink-3); font-size:.68rem; letter-spacing:.1em; text-transform:uppercase; padding:6px}
  .cm .cell{background:var(--surface-2); font-size:var(--step-1)}
  .cm .cell.hit{background:var(--ok-soft); color:var(--ok)}
  .cm .cell.miss{background:var(--stop-soft); color:var(--stop)}
  .cm-legend{font-size:var(--step--1); color:var(--ink-3); margin-top:10px}

  /* ---- filters + table ---- */
  .filters{display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin:18px 0 14px}
  .chip{background:var(--surface); border:1px solid var(--rule-strong); color:var(--ink-2); border-radius:2px;
    padding:7px 13px; font:inherit; font-size:var(--step--1); cursor:pointer}
  .chip:hover{border-color:var(--accent); color:var(--accent)}
  .chip[aria-pressed="true"]{background:var(--accent); border-color:var(--accent); color:var(--surface)}
  .chip:focus-visible{outline:2px solid var(--accent); outline-offset:2px}
  input[type=search]{background:var(--surface); border:1px solid var(--rule-strong); color:var(--ink);
    border-radius:2px; padding:7px 12px; font:inherit; font-size:var(--step--1); min-width:200px}
  input[type=search]:focus-visible{outline:2px solid var(--accent); outline-offset:1px; border-color:var(--accent)}
  .count{margin-left:auto; color:var(--ink-3); font-size:var(--step--1);
    font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums}
  .toolbar-actions{display:flex; gap:8px; align-items:center}
  .toolbar-actions .chip{padding:7px 10px}
  .view-summary{display:flex; gap:14px; align-items:center; flex-wrap:wrap;
    margin:0 0 14px; color:var(--ink-2); font-size:var(--step--1)}
  .view-summary strong{color:var(--ink); font-family:"IBM Plex Mono",monospace}

  .tablewrap{overflow-x:auto; border:1px solid var(--rule); border-radius:3px; background:var(--surface)}
  table{border-collapse:collapse; width:100%; font-size:var(--step--1)}
  th{position:sticky; top:0; background:var(--surface-2); text-align:left; padding:11px 14px;
    font-weight:600; font-size:.7rem; letter-spacing:.09em; text-transform:uppercase; color:var(--ink-2);
    cursor:pointer; white-space:nowrap; border-bottom:1px solid var(--rule-strong)}
  th:hover{color:var(--accent)}
  th[aria-sort]{color:var(--accent)}
  td{padding:10px 14px; border-top:1px solid var(--rule); white-space:nowrap}
  td.n{font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums; text-align:right}
  tbody tr:hover{background:var(--accent-soft)}
  .up{color:var(--ok)} .down{color:var(--stop)} .flat{color:var(--ink-3)}
  .empty{padding:44px; text-align:center; color:var(--ink-3)}

  .caveats{margin-top:18px; padding:18px 20px; background:var(--surface-2); border-radius:3px;
    border-left:3px solid var(--warn)}
  .caveats ul{margin:8px 0 0; padding-left:20px; color:var(--ink-2); font-size:var(--step--1)}
  .caveats li{margin:5px 0}
  footer{margin-top:64px; padding-top:22px; border-top:1px solid var(--rule); color:var(--ink-3); font-size:var(--step--1)}
  @media (prefers-reduced-motion: reduce){*{animation:none!important; transition:none!important}}
</style>

<div class="verdict">
  <div class="verdict-in">
    <span id="stamp" class="stamp"><span class="dot"></span><span id="stampText"></span></span>
    <h1>Pricing run review</h1>
    <div class="meta">
      <span class="num" id="asOf"></span>
      <button class="toggle" id="themeBtn" type="button">Theme</button>
    </div>
  </div>
</div>

<div class="wrap">
  <p class="eyebrow" style="margin-top:36px">Batch state</p>
  <h2 style="margin-top:0">Can this batch publish?</h2>
  <p class="sub">The circuit breakers decide before anything else does. They catch faults that belong to the
    batch rather than to any single price — every individual move can be legal and well scored while the run
    as a whole is a catastrophe.</p>
  <div class="card" id="breakers"></div>

  <p class="eyebrow">Decision quality</p>
  <h2 style="margin-top:0">Not just whether we answered</h2>
  <p class="sub">A system that always returns a price hides its own degradation inside a healthy-looking
    success rate. The rung histogram is what makes “we returned a price” and “we returned a good price”
    different numbers. Click a rung to filter everything below.</p>
  <div class="grid g2">
    <div class="card"><span class="label">Degradation rungs</span><div class="bars" id="rungBars"></div></div>
    <div class="card"><span class="label">Price movement</span><div class="bars" id="moveBars"></div>
      <p class="foot" id="moveFoot"></p></div>
  </div>

  <div class="grid g4" id="tiles" style="margin-top:16px"></div>

  <p class="eyebrow">Estimator health</p>
  <h2 style="margin-top:0">Elasticity, and whether to believe it</h2>
  <p class="sub">A positive estimated elasticity is almost always residual confounding rather than a Giffen
    good, so the sign-violation rate is a diagnostic on the estimator — not a finding about buyers.</p>
  <div class="grid g2">
    <div class="card"><span class="label">Distribution across SKUs</span><div id="elasticHist" class="bars"></div></div>
    <div class="card"><span class="label">Health checks</span><div id="elasticHealth"></div></div>
  </div>

  <p class="eyebrow">Model evaluation</p>
  <h2 style="margin-top:0">Where a confusion matrix belongs</h2>
  <p class="sub">Nothing in the pricing path emits a class label — demand is a quantile, elasticity a
    continuous parameter, retention a hazard. Forcing a confusion matrix onto any of them would mean inventing
    a threshold and scoring a distinction the model was never asked to make. The <em>circuit breaker</em> is a
    real binary classifier, so it gets a real matrix: scored against batches whose status is known by
    construction, with faults spanning subtle to obvious so the boundary is actually tested.</p>
  <div class="grid g2">
    <div class="card"><span class="label">Circuit breaker, on labelled batches</span>
      <div class="cm" id="cm"></div><p class="cm-legend" id="cmLegend"></p></div>
    <div class="card"><span class="label">Scores</span><div id="cmScores"></div></div>
  </div>

  <p class="eyebrow">Per-SKU detail</p>
  <h2 style="margin-top:0">Every decision, with its reason</h2>
  <p class="sub">Filters here and the rung bars above are the same filter — the useful question is rarely
    “what is the median” but “what does it look like among the ones that degraded”.</p>
  <div class="filters">
    <button class="chip" data-conf="high" type="button">Elasticity trusted</button>
    <button class="chip" data-conf="low" type="button">Elasticity not trusted</button>
    <button class="chip" data-dir="up" type="button">Price up</button>
    <button class="chip" data-dir="down" type="button">Price down</button>
    <button class="chip" id="clear" type="button">Clear</button>
    <button class="chip" id="copyView" type="button">Copy view link</button>
    <input type="search" id="q" placeholder="Search SKU" aria-label="Search SKU">
    <div class="toolbar-actions">
      <button class="chip" id="exportCsv" type="button">Export CSV</button>
      <button class="chip" id="exportJson" type="button">Export JSON</button>
    </div>
    <span class="count" id="count"></span>
  </div>
  <div class="view-summary" aria-live="polite">
    <span>Current view: <strong id="viewCount">0</strong> decisions</span>
    <span>Expected profit: <strong id="viewProfit">—</strong></span>
  </div>
  <div class="tablewrap"><table>
    <thead><tr>
      <th data-k="sku">SKU</th>
      <th data-k="current_price" class="n">Current</th>
      <th data-k="recommended_price" class="n">Recommended</th>
      <th data-k="price_change_pct" class="n">Change</th>
      <th data-k="elasticity" class="n">Elasticity</th>
      <th data-k="elasticity_confidence">Trust</th>
      <th data-k="degradation_rung" class="n">Rung</th>
      <th data-k="binding_constraints">Binding</th>
    </tr></thead>
    <tbody id="rows"></tbody>
  </table></div>

  <div class="caveats">
    <strong>What these numbers rest on</strong>
    <ul id="notes"></ul>
  </div>
  <footer>Generated by <span class="num">prismprice.observability.dashboard</span> from one pipeline run.
    Recommendations are decision support, not published prices.</footer>
</div>

<script>
const DATA = __PAYLOAD__;
const S = DATA.summary, D = DATA.decisions;
const state = {rung:null, conf:null, dir:null, q:"", sort:"price_change_pct", desc:true};
const stateKeys = ["rung","conf","dir","q","sort","desc"];

const pct = v => (v*100).toFixed(1) + "%";
const money = v => v==null ? "—" : v.toFixed(2);
const el = (t,c,x) => {const n=document.createElement(t); if(c)n.className=c; if(x!=null)n.textContent=x; return n;};
const download = (name, content, type) => {
  const blob = new Blob([content], {type});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob); link.download = name; link.click();
  URL.revokeObjectURL(link.href);
};
const csvCell = value => {
  const text = value == null ? "" : String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
};

/* theme */
const root = document.documentElement;
document.getElementById("themeBtn").onclick = () => {
  const dark = getComputedStyle(root).getPropertyValue("--ground").trim().toLowerCase().startsWith("#0e");
  root.setAttribute("data-theme", dark ? "light" : "dark");
};

/* verdict */
const stamp = document.getElementById("stamp"), stampText = document.getElementById("stampText");
stamp.classList.add(S.may_publish ? "go" : "stop");
stampText.textContent = S.may_publish ? "Cleared to publish" : "Publication halted";
document.getElementById("asOf").textContent = (S.as_of||"").slice(0,10);

/* breakers */
const bw = document.getElementById("breakers");
(S.breakers||[]).forEach(b => {
  const row = el("div","breaker");
  row.append(el("span","name", b.name.replace(/_/g," ")));
  row.append(el("span","pill " + (b.halted?"stop":"go"), b.halted?"HALT":"PASS"));
  row.append(el("span","why", b.detail));
  bw.append(row);
});

/* filtering */
function filtered(){
  return D.filter(d =>
    (state.rung===null || +d.degradation_rung===state.rung) &&
    (state.conf===null || d.elasticity_confidence===state.conf) &&
    (state.dir===null || (state.dir==="up" ? d.price_change_pct>1e-9 : d.price_change_pct<-1e-9)) &&
    (!state.q || String(d.sku).toLowerCase().includes(state.q))
  );
}

function syncHash(){
  const params = new URLSearchParams();
  stateKeys.forEach(key => { if(state[key] !== null && state[key] !== "") params.set(key, state[key]); });
  history.replaceState(null, "", params.toString() ? "#" + params.toString() : location.pathname);
}
function loadHash(){
  const params = new URLSearchParams(location.hash.slice(1));
  stateKeys.forEach(key => {
    if(!params.has(key)) return;
    const value=params.get(key);
    state[key]=key==="rung" ? Number(value) : key==="desc" ? value==="true" : value;
  });
  document.getElementById("q").value=state.q;
}

function bars(node, rows, total, onclick){
  node.replaceChildren();
  rows.forEach(r => {
    const b = el("button","bar-row"); b.type="button";
    if(onclick){ b.setAttribute("aria-pressed", String(r.active)); b.onclick = () => onclick(r); }
    else { b.style.cursor="default"; }
    b.append(el("span","k", r.label));
    const track = el("div","bar-track"), fill = el("div","bar-fill" + (r.tone?" is-"+r.tone:""));
    fill.style.width = (total ? (r.value/total*100) : 0).toFixed(1) + "%";
    track.append(fill); b.append(track);
    b.append(el("span","v", r.display != null ? r.display : String(r.value)));
    node.append(b);
  });
}

function tile(label, figure, foot, tone){
  const c = el("div","card stripe" + (tone?" is-"+tone:""));
  c.append(el("span","label",label));
  c.append(el("div","figure",figure));
  if(foot) c.append(el("p","foot",foot));
  return c;
}

function render(){
  const rows = filtered();
  const n = rows.length;
  syncHash();
  const profit = rows.reduce((total, row) => total + (Number(row.expected_profit) || 0), 0);
  document.getElementById("viewCount").textContent = String(n);
  document.getElementById("viewProfit").textContent = money(profit);

  /* rung histogram — always over the whole run, with the active one marked */
  const rungLabels = {1:"1 · optimal",2:"2 · stale feed",3:"3 · pooled β",4:"4 · rule engine",5:"5 · held"};
  const counts = S.rung_counts;
  bars(document.getElementById("rungBars"),
    [1,2,3,4,5].map(r => ({
      label: rungLabels[r], value: counts[String(r)]||0,
      display: String(counts[String(r)]||0), active: state.rung===r,
      tone: r>=4 ? "stop" : (r===3 ? "warn" : null)
    })), S.n_priced, r => {
      const which = +r.label.split(" ")[0];
      state.rung = state.rung===which ? null : which; render();
    });

  const up = rows.filter(d=>d.price_change_pct>1e-9).length;
  const down = rows.filter(d=>d.price_change_pct<-1e-9).length;
  const flat = n-up-down;
  bars(document.getElementById("moveBars"), [
    {label:"Increase", value:up, display:String(up), tone:null},
    {label:"Decrease", value:down, display:String(down), tone:"stop"},
    {label:"No change", value:flat, display:String(flat), tone:"warn"},
  ], n||1, null);
  const med = rows.length ? rows.map(d=>d.price_change_pct).sort((a,b)=>a-b)[Math.floor(rows.length/2)] : 0;
  document.getElementById("moveFoot").textContent =
    "Median change " + (med>=0?"+":"") + pct(med) + " across " + n + " decisions in view.";

  /* tiles recompute against the filter */
  const es = rows.map(d=>d.elasticity).filter(v=>v!=null && isFinite(v)).sort((a,b)=>a-b);
  const medE = es.length ? es[Math.floor(es.length/2)] : NaN;
  const trusted = rows.filter(d=>d.elasticity_confidence==="high").length;
  const degraded = rows.filter(d=>+d.degradation_rung>=3).length;
  const t = document.getElementById("tiles"); t.replaceChildren();
  t.append(tile("SKUs in view", String(n), S.n_priced + " priced in the run"));
  t.append(tile("Median elasticity", isFinite(medE)? medE.toFixed(2) : "—",
    es.length + " identified", isFinite(medE) && medE>=0 ? "stop" : null));
  t.append(tile("Elasticity trusted", n? pct(trusted/n):"—",
    trusted + " of " + n + " usable", trusted/Math.max(n,1) < .6 ? "warn" : "ok"));
  t.append(tile("Degraded (rung ≥3)", n? pct(degraded/n):"—",
    degraded + " decisions", degraded/Math.max(n,1) > .05 ? "warn" : "ok"));

  /* table */
  const tb = document.getElementById("rows"); tb.replaceChildren();
  document.getElementById("count").textContent = n + " of " + D.length + " decisions";
  if(!n){ const tr=el("tr"); const td=el("td","empty","No decisions match these filters."); td.colSpan=8; tr.append(td); tb.append(tr); return; }
  const sorted = rows.slice().sort((a,b)=>{
    const x=a[state.sort], y=b[state.sort];
    if(typeof x==="string") return state.desc ? String(y).localeCompare(String(x)) : String(x).localeCompare(String(y));
    return state.desc ? (y??0)-(x??0) : (x??0)-(y??0);
  });
  sorted.slice(0,400).forEach(d=>{
    const tr=el("tr");
    tr.append(el("td",null,d.sku));
    tr.append(el("td","n",money(d.current_price)));
    tr.append(el("td","n",money(d.recommended_price)));
    const ch=el("td","n " + (d.price_change_pct>1e-9?"up":d.price_change_pct<-1e-9?"down":"flat"),
      (d.price_change_pct>=0?"+":"")+pct(d.price_change_pct));
    tr.append(ch);
    tr.append(el("td","n", d.elasticity!=null&&isFinite(d.elasticity)? d.elasticity.toFixed(2):"—"));
    const conf=el("td"); conf.append(el("span","pill "+(d.elasticity_confidence==="high"?"go":"mute"),
      d.elasticity_confidence==="high"?"HIGH":"LOW")); tr.append(conf);
    tr.append(el("td","n", String(d.degradation_rung)));
    tr.append(el("td",null,(d.binding_constraints||[]).join(", ")||"—"));
    tb.append(tr);
  });
}

/* elasticity histogram (fixed, over the run) */
(function(){
  const pts = Object.values(DATA.elasticities).map(e=>e.point).filter(v=>v!=null&&isFinite(v));
  const edges=[-4,-3,-2,-1.5,-1,-0.5,0,1];
  const rows=[];
  for(let i=0;i<edges.length-1;i++){
    const c=pts.filter(v=>v>=edges[i]&&v<edges[i+1]).length;
    rows.push({label:edges[i].toFixed(1)+" to "+edges[i+1].toFixed(1), value:c, display:String(c),
      tone: edges[i]>=0 ? "stop" : null});
  }
  bars(document.getElementById("elasticHist"), rows, Math.max(pts.length,1), null);
})();

/* elasticity health */
(function(){
  const h=S.elasticity.health, box=document.getElementById("elasticHealth");
  const line=(k,v,tone)=>{const r=el("div","breaker");
    r.append(el("span","name",k)); r.append(el("span","pill "+tone, v)); return r;};
  box.append(line("Sign violations", pct(h.sign_violation_rate||0),
    h.sign_violations_breached ? "stop":"go"));
  box.append(line("Median CI width", (h.median_ci_width??NaN).toFixed(2),
    h.ci_width_breached ? "warn":"go"));
  box.append(line("Declined to answer", String(h.n_declined??0), (h.n_declined||0)>0 ? "warn":"go"));
  box.append(line("Overall", h.healthy ? "HEALTHY":"NOT HEALTHY", h.healthy?"go":"stop"));
})();

/* confusion matrix */
(function(){
  const c=S.classification; const grid=document.getElementById("cm");
  if(!c){ grid.append(el("div","cm-legend","No decisions to score.")); return; }
  const g=c.grid;
  const cells=[["","Published","Halted"],
               ["Good batch", g[0][0], g[0][1]],
               ["Bad batch",  g[1][0], g[1][1]]];
  cells.forEach((row,ri)=>row.forEach((v,ci)=>{
    const isHead = ri===0 || ci===0;
    const d=el("div", isHead?"hd":"cell", String(v));
    if(!isHead) d.classList.add((ri===ci)?"hit":"miss");
    grid.append(d);
  }));
  document.getElementById("cmLegend").textContent =
    "Rows: what the batch really was. Columns: what the breakers did. "
    + "Top-right is a good run held up; bottom-left is a bad one published.";
  const s=document.getElementById("cmScores");
  const line=(k,v,tone)=>{const r=el("div","breaker");
    r.append(el("span","name",k)); r.append(el("span","pill "+tone, v)); return r;};
  s.append(line("Precision", isFinite(c.precision)? c.precision.toFixed(3):"—","mute"));
  s.append(line("Recall", isFinite(c.recall)? c.recall.toFixed(3):"—","mute"));
  s.append(line("Specificity", isFinite(c.specificity)? c.specificity.toFixed(3):"—","mute"));
  s.append(line("Balanced accuracy", isFinite(c.balanced_accuracy)? c.balanced_accuracy.toFixed(3):"—","mute"));
  if(c.accuracy_is_misleading){
    const w=el("p","foot","Classes are imbalanced here, so plain accuracy ("
      + c.accuracy.toFixed(3) + ") flatters. Read recall and specificity instead.");
    s.append(w);
  }
})();

/* notes */
(function(){
  const ul=document.getElementById("notes");
  (S.notes||[]).forEach(n=>ul.append(el("li",null,n)));
  if((S.unobserved||[]).length)
    ul.append(el("li",null,"Unobserved confounders that could not be controlled for: "
      + S.unobserved.join(", ") + "."));
})();

/* controls */
document.querySelectorAll(".chip[data-conf]").forEach(b=>b.onclick=()=>{
  state.conf = state.conf===b.dataset.conf ? null : b.dataset.conf;
  document.querySelectorAll(".chip[data-conf]").forEach(x=>x.setAttribute("aria-pressed",String(x.dataset.conf===state.conf)));
  render();
});
document.querySelectorAll(".chip[data-dir]").forEach(b=>b.onclick=()=>{
  state.dir = state.dir===b.dataset.dir ? null : b.dataset.dir;
  document.querySelectorAll(".chip[data-dir]").forEach(x=>x.setAttribute("aria-pressed",String(x.dataset.dir===state.dir)));
  render();
});
document.getElementById("clear").onclick=()=>{
  state.rung=state.conf=state.dir=null; state.q=""; document.getElementById("q").value="";
  document.querySelectorAll(".chip[data-conf],.chip[data-dir]").forEach(x=>x.setAttribute("aria-pressed","false"));
  render();
};
document.getElementById("copyView").onclick=async()=>{
  syncHash();
  await navigator.clipboard?.writeText(location.href);
  document.getElementById("copyView").textContent="View link copied";
  setTimeout(()=>document.getElementById("copyView").textContent="Copy view link", 1400);
};
document.getElementById("q").oninput=e=>{state.q=e.target.value.toLowerCase().trim(); render();};
document.getElementById("exportCsv").onclick=()=>{
  const columns=["sku","current_price","recommended_price","price_change_pct",
    "elasticity","elasticity_confidence","degradation_rung","expected_profit"];
  const lines=[columns.join(","), ...filtered().map(row=>columns.map(key=>csvCell(row[key])).join(","))];
  download("prismprice-decisions.csv", lines.join("\n"), "text/csv;charset=utf-8");
};
document.getElementById("exportJson").onclick=()=>{
  download("prismprice-decisions.json", JSON.stringify(filtered(), null, 2), "application/json");
};
document.querySelectorAll("th[data-k]").forEach(th=>th.onclick=()=>{
  const k=th.dataset.k;
  if(state.sort===k) state.desc=!state.desc; else {state.sort=k; state.desc=true;}
  document.querySelectorAll("th[data-k]").forEach(x=>x.removeAttribute("aria-sort"));
  th.setAttribute("aria-sort", state.desc?"descending":"ascending");
  render();
});

render();
loadHash();
document.querySelectorAll(".chip[data-conf]").forEach(x=>x.setAttribute("aria-pressed",String(x.dataset.conf===state.conf)));
document.querySelectorAll(".chip[data-dir]").forEach(x=>x.setAttribute("aria-pressed",String(x.dataset.dir===state.dir)));
render();
document.addEventListener("keydown", event=>{
  if(event.key==="/" && document.activeElement.tagName!=="INPUT"){
    event.preventDefault(); document.getElementById("q").focus();
  }
  if(event.key==="Escape" && document.activeElement===document.getElementById("q")){
    document.getElementById("clear").click();
  }
});
</script>
"""
