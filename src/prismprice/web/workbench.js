import { buildRequest, fetchHealth, requestDecision } from "./service.js";
import { renderDecision } from "./render.js";

const form = document.querySelector("#decision-form");
const submit = document.querySelector("#submit-button");
const error = document.querySelector("#form-error");
const modelLabel = document.querySelector("#model-label");
const result = document.querySelector("#result-content");
const empty = document.querySelector("#result-empty");
const panel = document.querySelector("#decision-panel");
const serviceState = document.querySelector("#service-state");
const serviceLabel = document.querySelector("#service-label");
const retryButton = document.querySelector("#service-retry");
const staleNotice = document.querySelector("#result-stale");
const promoToggle = form.elements.namedItem("is_promo");
const promoField = form.elements.namedItem("min_price_last_30d");
let hasDecision = false;
let healthSequence = 0;

function decisionInputsSnapshot() {
  return JSON.stringify([...new FormData(form).entries()].filter(([name]) => name !== "api_key"));
}

export async function refreshServiceState() {
  const sequence = ++healthSequence;
  serviceState.className = "service-state loading";
  serviceLabel.textContent = "Connecting";
  retryButton.hidden = true;
  retryButton.disabled = true;
  try {
    const health = await fetchHealth();
    if (sequence !== healthSequence) return health;
    const demo = String(health.policy_version).toLowerCase().startsWith("demo");
    const available = health.model_store_available;
    serviceState.className = `service-state${available ? "" : " offline"}`;
    serviceLabel.textContent = available ? (demo ? "Demo service online" : "Pricing API online") : "Model store unavailable";
    modelLabel.textContent = !available
      ? "Model store unavailable · decisions will hold current prices"
      : demo ? "Sample model · demo only" : `Policy ${health.policy_version} · connected`;
    modelLabel.dataset.mode = available ? (demo ? "demo" : "live") : "offline";
    retryButton.hidden = available;
    return health;
  } catch {
    if (sequence !== healthSequence) return null;
    serviceState.className = "service-state offline";
    serviceLabel.textContent = "Service unreachable";
    modelLabel.textContent = "Start the local PrismPrice server";
    modelLabel.dataset.mode = "offline";
    retryButton.hidden = false;
    return null;
  } finally {
    if (sequence === healthSequence) retryButton.disabled = false;
  }
}

function showError(message) {
  error.textContent = message;
  error.hidden = false;
  error.focus();
}

function showLoading() {
  result.replaceChildren();
  const card = document.createElement("div");
  card.className = "service-error";
  const title = document.createElement("h3");
  title.textContent = "Scoring the candidate ladder…";
  const note = document.createElement("p");
  note.textContent = "Checking the model response and every price guardrail.";
  card.append(title, note);
  result.append(card);
  empty.hidden = true;
  result.hidden = false;
  staleNotice.hidden = true;
}

async function submitDecision(event) {
  event.preventDefault();
  error.hidden = true;
  let body;
  try {
    body = buildRequest(form);
  } catch (problem) {
    showError(problem.message);
    return;
  }
  const submittedInputs = decisionInputsSnapshot();

  submit.disabled = true;
  submit.setAttribute("aria-busy", "true");
  submit.textContent = "Scoring the price ladder…";
  hasDecision = false;
  showLoading();
  try {
    const apiKey = form.elements.namedItem("api_key").value;
    const decision = await requestDecision(body, apiKey);
    renderDecision(decision, result);
    hasDecision = true;
    staleNotice.hidden = decisionInputsSnapshot() === submittedInputs;
    panel.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "nearest" });
  } catch (problem) {
    result.replaceChildren();
    const card = document.createElement("div");
    card.className = "service-error";
    const title = document.createElement("h3");
    title.textContent = "Decision could not be completed.";
    const note = document.createElement("p");
    note.textContent = problem instanceof Error ? problem.message : "An unexpected error prevented this decision.";
    card.append(title, note);
    result.append(card);
    showError(note.textContent);
  } finally {
    result.hidden = false;
    empty.hidden = true;
    submit.disabled = false;
    submit.removeAttribute("aria-busy");
    submit.textContent = "Find a safe price ↗";
  }
}

form.addEventListener("submit", submitDecision);
promoToggle.addEventListener("change", () => {
  form.querySelector(".promo-anchor").hidden = !promoToggle.checked;
  promoField.required = promoToggle.checked;
  if (promoToggle.checked) promoField.focus();
});

function markDecisionStale(event) {
  error.hidden = true;
  if (hasDecision && event.target.name !== "api_key") staleNotice.hidden = false;
}

form.addEventListener("input", markDecisionStale);
form.addEventListener("change", markDecisionStale);
retryButton.addEventListener("click", refreshServiceState);
window.addEventListener("online", refreshServiceState);
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) refreshServiceState();
});

refreshServiceState();
