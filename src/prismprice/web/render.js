const currency = document.documentElement.dataset.currency || "GBP";
const locale = currency === "GBP" ? "en-GB" : "en-US";
const money = (value) => new Intl.NumberFormat(locale, {
  style: "currency", currency, minimumFractionDigits: 2, maximumFractionDigits: 2,
}).format(value);
const number = (value, digits = 1) => new Intl.NumberFormat("en-US", {
  maximumFractionDigits: digits,
}).format(value);
const element = (tag, className, text) => {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text !== undefined && text !== null) item.textContent = text;
  return item;
};
const rungNames = {
  OK_OPTIMAL: ["Scenario scored · candidate selected", ""],
  WARN_STALE_COMPETITOR: ["Competitor feed is stale", "warn"],
  FALLBACK_BACKUP_MODEL: ["Validated backup model used", "warn"],
  FALLBACK_POOLED_ELASTICITY: ["Using pooled elasticity", "warn"],
  FALLBACK_RULE_ENGINE: ["Price held by guardrails", "stop"],
  CRITICAL_MAINTAIN_PREV: ["Safety hold · current price retained", "stop"],
};

function candidateEvidenceChart(record, target) {
  const candidates = [...record.candidates].sort((a, b) => a.price - b.price);
  const ranges = candidates.map((candidate) => candidate.demand_quantiles).filter(Boolean);
  const low = ranges.length ? Math.min(...ranges.map((range) => range.p10)) : 0;
  const high = ranges.length ? Math.max(...ranges.map((range) => range.p90)) : 1;
  const span = Math.max(high - low, 1e-9);
  const isHold = record.degradation_reason_code === "CRITICAL_MAINTAIN_PREV";
  const heading = element("div", "result-block-heading");
  heading.append(element("h4", "", isHold ? "Safety hold" : "Candidate trade-offs"));
  heading.append(element("span", "", isHold ? "No candidate prices were scored" : `${candidates.length} prices \u00b7 demand interval and contribution`));
  target.append(heading);
  if (isHold) {
    const reason = record.degradation_detail || "A validated demand forecast was unavailable.";
    target.append(element(
      "p",
      "result-warning",
      `${reason} PrismPrice retained the current price supplied in this request; no alternatives were scored. Confirm that it matches the live price before acting.`,
    ));
    return;
  }
  const columns = element("div", "candidate-table-heading");
  ["Price", "Demand P10 · P50 · P90", "Contribution / day · J", "Tail CVaR / day", "Rule"].forEach((label) => columns.append(element("span", "", label)));
  target.append(columns);
  const list = element("div", "candidate-list");
  if (!candidates.length) {
    list.append(element("p", "result-warning", record.degradation_detail || "No candidates were scored; the current price was held."));
  }
  candidates.forEach((candidate) => {
    const row = element("div", `candidate-row${candidate.price === record.recommended_price ? " selected" : ""}`);
    row.append(element("span", "candidate-price", money(candidate.price)));
    const demand = element("div", "candidate-demand");
    const quantiles = candidate.demand_quantiles;
    if (quantiles) {
      demand.append(element("span", "candidate-demand-label", `P10 ${number(quantiles.p10)} · P50 ${number(quantiles.p50)} · P90 ${number(quantiles.p90)}`));
      const track = element("div", "candidate-track");
      const interval = element("span", "candidate-interval");
      const start = ((quantiles.p10 - low) / span) * 100;
      const end = ((quantiles.p90 - low) / span) * 100;
      interval.style.left = `${start}%`;
      interval.style.width = `${Math.max(end - start, 1)}%`;
      const median = element("i", "candidate-median");
      median.style.left = `${((quantiles.p50 - low) / span) * 100}%`;
      track.setAttribute("role", "img");
      track.setAttribute("aria-label", `Demand interval ${number(quantiles.p10)} to ${number(quantiles.p90)} active-day units, median ${number(quantiles.p50)}`);
      track.append(interval, median);
      demand.append(track);
    } else {
      demand.append(element("span", "candidate-demand-label", "Forecast unavailable"));
    }
    row.append(demand);
    const contribution = element("span", "candidate-score");
    contribution.append(
      element("strong", "", candidate.expected_profit == null ? "—" : money(candidate.expected_profit)),
      element("small", "", `J ${money(candidate.j_score)}`),
    );
    contribution.title = "Expected contribution under the supplied unit-cost assumption.";
    row.append(contribution);
    const risk = element("span", "candidate-risk", money(candidate.cvar));
    risk.title = `Mean contribution in the worst ${number(candidate.cvar_alpha * 100)}% of simulated outcomes.`;
    row.append(risk);
    row.append(element("span", `candidate-state${candidate.feasible ? "" : " blocked"}`, candidate.feasible ? "Eligible" : "Blocked"));
    list.append(row);
  });
  target.append(list);
}

function renderModelEvidence(record, target) {
  const evidence = record.model_evidence;
  if (!evidence) {
    if (record.degradation_detail && record.degradation_reason_code !== "CRITICAL_MAINTAIN_PREV") {
      target.append(element("p", "result-warning", record.degradation_detail));
    }
    return;
  }
  const section = element("section", "model-evidence");
  const heading = element("div", "result-block-heading");
  heading.append(element("h4", "", "Forecast evidence"));
  heading.append(element(
    "span",
    "",
    evidence.model_role === "backup"
      ? "Validated backup model used"
      : evidence.validation_calibrated ? "Temporal calibration gate passed" : "Calibration gate not passed",
  ));
  section.append(heading);
  const metrics = element("div", "evidence-metrics");
  [
    ["Held-out WAPE", `${number(evidence.validation_wape * 100)}%`],
    ["80% interval coverage", `${number(evidence.validation_coverage_80 * 100)}%`],
    ["Validation rows", number(evidence.validation_rows, 0)],
  ].forEach(([label, value]) => {
    const item = element("div", "evidence-metric");
    item.append(element("span", "", label), element("strong", "", value));
    metrics.append(item);
  });
  section.append(metrics);
  section.append(element("p", "evidence-detail",
    `${evidence.model_name} ${evidence.model_version} · ${evidence.dataset_name} · trained through ${evidence.training_through.slice(0, 10)}. `
    + `${evidence.observations_used} active sales observations for this SKU; demand is measured in ${evidence.demand_unit}. `
    + `Held-out P10 tail coverage ${number(evidence.validation_p10_coverage * 100)}%; model digest ${evidence.artifact_sha256?.slice(0, 16) || "not recorded"}; `
    + `feature snapshot ${evidence.feature_snapshot_id.slice(0, 19)}…`));
  if (evidence.model_role === "backup" && record.degradation_detail) {
    section.append(element("p", "result-warning", record.degradation_detail));
  }
  target.append(section);
}

function guardrailList(record, target) {
  const checks = record.guardrail_results || [];
  const passed = checks.filter((check) => check.status === "PASSED").length;
  const failed = checks.filter((check) => check.status === "FAILED").length;
  const heading = element("div", "result-block-heading guardrail-heading");
  heading.append(element("h4", "", "Guardrail verdicts"));
  heading.append(element("span", "", checks.length ? `${passed} passed · ${failed} blocked` : "No checks returned"));
  target.append(heading);
  checks.forEach((check) => {
    const state = check.status === "FAILED" ? "failed" : check.status === "NOT_APPLICABLE" ? "na" : "";
    const row = element("div", `guardrail-row ${state}`);
    row.append(element("span", "guardrail-mark", check.status === "FAILED" ? "!" : check.status === "NOT_APPLICABLE" ? "–" : "✓"));
    row.append(element("span", "guardrail-name", check.constraint_name.replaceAll("_", " ")));
    row.append(element("span", "guardrail-detail", check.detail));
    target.append(row);
  });
}

export function renderDecision(payload, target) {
  const record = validatedRecord(payload);
  const code = record.degradation_reason_code;
  const isHold = code === "CRITICAL_MAINTAIN_PREV";
  const [verdict, tone] = rungNames[code] || [code.replaceAll("_", " "), "warn"];
  const change = (record.recommended_price / record.inputs.current_price) - 1;
  target.replaceChildren();

  const header = element("div", "result-header");
  header.append(element("span", `verdict-pill ${tone}`, verdict));
  header.append(element("span", "result-meta", `Rung ${record.degradation_rung} of 5`));
  target.append(header);

  const main = element("div", "recommendation-block");
  const price = element("div", "");
  price.append(element("span", "recommendation-label", `${isHold ? "Current price retained" : "Selected candidate"} \u00b7 ${record.sku}`));
  price.append(element("strong", "recommendation-price", money(record.recommended_price)));
  main.append(price);
  const deltaText = isHold ? "No change" : `${change > 0 ? "\u2191" : change < 0 ? "\u2193" : "\u2194"} ${number(Math.abs(change * 100))}%`;
  main.append(element("span", `recommendation-delta ${isHold || change === 0 ? "flat" : change < 0 ? "down" : ""}`, deltaText));
  target.append(main);

  const estimates = record.estimates?.demand_quantiles;
  const stats = element("div", "result-stats");
  const statRows = isHold ? [
    ["Current price", money(record.inputs.current_price)],
    ["Demand estimate", "Unavailable"],
    ["Candidates scored", "0"],
  ] : [
    ["Demand range", estimates ? `${number(estimates.p10)}\u2013${number(estimates.p90)} active-day units` : "Not available"],
    ["Median demand", estimates ? `${number(estimates.p50)} active-day units` : "Not available"],
    ["Contribution / active day", record.candidates.find((candidate) => candidate.price === record.recommended_price)?.expected_profit == null
      ? "Not available" : money(record.candidates.find((candidate) => candidate.price === record.recommended_price).expected_profit)],
  ];
  statRows.forEach(([label, value]) => {
    const stat = element("div", "result-stat");
    stat.append(element("span", "", label), element("strong", "", value));
    stats.append(stat);
  });
  target.append(stats);

  renderModelEvidence(record, target);
  const candidates = element("section", "candidate-ladder");
  candidateEvidenceChart(record, candidates);
  target.append(candidates);
  if (!isHold || record.guardrail_results?.length) {
    const checks = element("section", "guardrail-results");
    guardrailList(record, checks);
    target.append(checks);
  }

  const audit = element("div", "audit-trail");
  const auditId = element("span", "", `Decision · ${record.decision_id}`);
  auditId.title = record.decision_id;
  audit.append(auditId, element("span", "", `Policy · ${record.policy_version}`), element("span", "", `Seed · ${record.seed}`));
  const exportButton = element("button", "record-export", "Download audit record ↗");
  exportButton.type = "button";
  exportButton.addEventListener("click", () => downloadRecord(record));
  audit.append(exportButton);
  target.append(audit);
}

function downloadRecord(record) {
  const blob = new Blob([JSON.stringify(record, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = element("a", "", "");
  link.href = url;
  const safeSku = record.sku.replace(/[^a-z0-9._-]+/gi, "-").replace(/^-+|-+$/g, "").slice(0, 48) || "item";
  link.download = `prismprice-${safeSku}-${record.decision_id.slice(0, 8)}.json`;
  link.hidden = true;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function validatedRecord(payload) {
  const record = payload?.record;
  const validNumber = (value) => typeof value === "number" && Number.isFinite(value);
  if (!record || typeof record !== "object"
    || typeof record.sku !== "string"
    || typeof record.decision_id !== "string"
    || typeof record.degradation_reason_code !== "string"
    || !Number.isInteger(record.degradation_rung) || record.degradation_rung < 1 || record.degradation_rung > 5
    || !record.inputs || !validNumber(record.inputs.current_price) || record.inputs.current_price <= 0
    || !validNumber(record.recommended_price) || record.recommended_price <= 0
    || !Array.isArray(record.candidates) || !Array.isArray(record.guardrail_results)
    || typeof record.policy_version !== "string" || !validNumber(record.seed)) {
    throw new Error("The pricing API response is missing required decision details. No recommendation was displayed.");
  }
  if (!record.candidates.every((candidate) => candidate
    && validNumber(candidate.price) && candidate.price > 0
    && validNumber(candidate.j_score) && validNumber(candidate.cvar)
    && typeof candidate.feasible === "boolean")) {
    throw new Error("The pricing API returned an incomplete candidate ladder. No recommendation was displayed.");
  }
  if (!record.guardrail_results.every((check) => check
    && typeof check.constraint_name === "string"
    && typeof check.detail === "string"
    && ["PASSED", "FAILED", "NOT_APPLICABLE"].includes(check.status))) {
    throw new Error("The pricing API returned incomplete guardrail results. No recommendation was displayed.");
  }
  const quantiles = record.estimates?.demand_quantiles;
  if (quantiles && ![quantiles.p10, quantiles.p50, quantiles.p90].every(validNumber)) {
    throw new Error("The pricing API returned invalid demand estimates. No recommendation was displayed.");
  }
  return record;
}
