import { buildRequest, fetchHealth, requestDecision } from "./service.js";
import { renderDecision } from "./render.js";

const form = document.querySelector("#decision-form");
const submit = document.querySelector("#submit-button");
const error = document.querySelector("#form-error");
const modelLabel = document.querySelector("#model-label");
const result = document.querySelector("#result-content");
const empty = document.querySelector("#result-empty");
const panel = document.querySelector("#decision-panel");

export async function refreshServiceState() {
  const state = document.querySelector("#service-state");
  state.className = "service-state loading";
  state.lastChild.textContent = "Connecting";
  try {
    const health = await fetchHealth();
    const demo = String(health.policy_version).toLowerCase().startsWith("demo");
    state.className = `service-state${health.model_store_available ? "" : " offline"}`;
    state.lastChild.textContent = health.model_store_available ? (demo ? "Demo service online" : "Pricing API online") : "Model store unavailable";
    modelLabel.textContent = demo ? "Sample model · demo only" : `Policy ${health.policy_version} · connected`;
    modelLabel.dataset.mode = demo ? "demo" : "live";
    return health;
  } catch {
    state.className = "service-state offline";
    state.lastChild.textContent = "Service unreachable";
    modelLabel.textContent = "Start the local PrismPrice server";
    modelLabel.dataset.mode = "offline";
    return null;
  }
}

function showError(message) {
  error.textContent = message;
  error.hidden = false;
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

  submit.disabled = true;
  submit.textContent = "Scoring the price ladder…";
  showLoading();
  try {
    const decision = await requestDecision(body);
    renderDecision(decision, result);
    panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (problem) {
    result.replaceChildren();
    const card = document.createElement("div");
    card.className = "service-error";
    const title = document.createElement("h3");
    title.textContent = "Decision could not be completed.";
    const note = document.createElement("p");
    note.textContent = problem.message;
    card.append(title, note);
    result.append(card);
    showError(problem.message);
  } finally {
    result.hidden = false;
    empty.hidden = true;
    submit.disabled = false;
    submit.textContent = "Find a safe price ↗";
  }
}

form.addEventListener("submit", submitDecision);
form.elements.namedItem("is_promo").addEventListener("change", (event) => {
  form.querySelector(".promo-anchor").hidden = !event.target.checked;
});

refreshServiceState();
