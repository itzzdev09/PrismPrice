const money = (value) => new Intl.NumberFormat("en-US", {
  style: "currency", currency: "USD", minimumFractionDigits: 2, maximumFractionDigits: 2,
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
  OK_OPTIMAL: ["Recommendation ready", ""],
  WARN_STALE_COMPETITOR: ["Competitor feed is stale", "warn"],
  FALLBACK_POOLED_ELASTICITY: ["Using pooled elasticity", "warn"],
  FALLBACK_RULE_ENGINE: ["Price held by guardrails", "stop"],
  CRITICAL_MAINTAIN_PREV: ["Model unavailable · price held", "stop"],
};

function candidateChart(record, target) {
  const candidates = [...(record.candidates || [])].sort((a, b) => a.price - b.price);
  const heading = element("div", "result-block-heading");
  heading.append(element("h4", "", "Candidate ladder"));
  heading.append(element("span", "", `${candidates.length} prices scored`));
  target.append(heading);
  const list = element("div", "candidate-list");
  if (!candidates.length) {
    list.append(element("p", "result-warning", "No candidates were scored. PrismPrice kept the current price because the decision model was unavailable."));
  }
  const scores = candidates.map((candidate) => candidate.j_score);
  const low = Math.min(...scores), high = Math.max(...scores);
  candidates.forEach((candidate) => {
    const row = element("div", `candidate-row${candidate.price === record.recommended_price ? " selected" : ""}`);
    row.append(element("span", "candidate-price", money(candidate.price)));
    const track = element("div", "candidate-track");
    const fill = element("div", "candidate-fill");
    const width = high === low ? 82 : 18 + ((candidate.j_score - low) / (high - low)) * 76;
    fill.style.width = `${width}%`;
    track.append(fill);
    row.append(track);
    row.append(element("span", `candidate-state${candidate.feasible ? "" : " blocked"}`, candidate.feasible ? "Eligible" : "Blocked"));
    list.append(row);
  });
  target.append(list);
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
  const record = payload.record;
  const code = record.degradation_reason_code;
  const [verdict, tone] = rungNames[code] || [code.replaceAll("_", " "), "warn"];
  const change = (record.recommended_price / record.inputs.current_price) - 1;
  target.replaceChildren();

  const header = element("div", "result-header");
  header.append(element("span", `verdict-pill ${tone}`, verdict));
  header.append(element("span", "result-meta", `Rung ${record.degradation_rung} of 5`));
  target.append(header);

  const main = element("div", "recommendation-block");
  const price = element("div", "");
  price.append(element("span", "recommendation-label", `Recommended price · ${record.sku}`));
  price.append(element("strong", "recommendation-price", money(record.recommended_price)));
  main.append(price);
  const deltaText = `${change > 0 ? "↑" : change < 0 ? "↓" : "↔"} ${number(Math.abs(change * 100))}%`;
  main.append(element("span", `recommendation-delta ${change === 0 ? "flat" : change < 0 ? "down" : ""}`, deltaText));
  target.append(main);

  const estimates = record.estimates?.demand_quantiles;
  const stats = element("div", "result-stats");
  const statRows = [
    ["Demand range", estimates ? `${number(estimates.p10)}–${number(estimates.p90)} units` : "Not available"],
    ["Median demand", estimates ? `${number(estimates.p50)} units` : "Not available"],
    ["Policy version", record.policy_version || "—"],
  ];
  statRows.forEach(([label, value]) => {
    const stat = element("div", "result-stat");
    stat.append(element("span", "", label), element("strong", "", value));
    stats.append(stat);
  });
  target.append(stats);

  const candidates = element("section", "candidate-ladder");
  candidateChart(record, candidates);
  target.append(candidates);
  const checks = element("section", "guardrail-results");
  guardrailList(record, checks);
  target.append(checks);

  const audit = element("div", "audit-trail");
  const auditId = element("span", "", `Decision · ${record.decision_id}`);
  auditId.title = record.decision_id;
  audit.append(auditId, element("span", "", `Seed · ${record.seed}`));
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
  link.download = `prismprice-${record.sku}-${record.decision_id.slice(0, 8)}.json`;
  link.click();
  URL.revokeObjectURL(url);
}
