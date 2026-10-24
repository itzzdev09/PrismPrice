import { buildDecisionRequest } from "./decision-request.js";
import { createPricingApi } from "./pricing-api.js";
import { renderDecision } from "./render.js";

export function initWorkbench(documentRoot = document, api = createPricingApi()) {
  const form = documentRoot.querySelector("#decision-form");
  if (!form) return null;

  const view = documentRoot.defaultView || window;
  const submit = documentRoot.querySelector("#submit-button");
  const submitLabel = submit.querySelector(".submit-label");
  const error = documentRoot.querySelector("#form-error");
  const modelLabel = documentRoot.querySelector("#model-label");
  const result = documentRoot.querySelector("#result-content");
  const empty = documentRoot.querySelector("#result-empty");
  const panel = documentRoot.querySelector("#decision-panel");
  const serviceState = documentRoot.querySelector("#service-state");
  const serviceLabel = documentRoot.querySelector("#service-label");
  const retryButton = documentRoot.querySelector("#service-retry");
  const staleNotice = documentRoot.querySelector("#result-stale");
  const promoToggle = form.elements.namedItem("is_promo");
  const promoField = form.elements.namedItem("min_price_last_30d");
  let hasDecision = false;
  let healthSequence = 0;

  const inputsSnapshot = () => JSON.stringify(
    [...new FormData(form).entries()].filter(([name]) => name !== "api_key"),
  );

  async function refreshServiceState() {
    const sequence = ++healthSequence;
    serviceState.className = "service-state loading";
    serviceLabel.textContent = "Connecting";
    retryButton.hidden = true;
    retryButton.disabled = true;
    try {
      const health = await api.fetchHealth();
      if (sequence !== healthSequence) return health;
      const demo = health.policy_version.toLowerCase().startsWith("demo");
      const available = health.model_store_available;
      serviceState.className = `service-state${available ? "" : " offline"}`;
      serviceLabel.textContent = available
        ? (demo ? "Demo service online" : "Pricing API online")
        : "Model store unavailable";
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

  function showResultMessage(titleText, description) {
    const card = document.createElement("div");
    card.className = "service-error";
    const title = document.createElement("h3");
    title.textContent = titleText;
    const note = document.createElement("p");
    note.textContent = description;
    card.append(title, note);
    result.replaceChildren(card);
    empty.hidden = true;
    result.hidden = false;
    staleNotice.hidden = true;
    return note;
  }

  async function submitDecision(event) {
    event.preventDefault();
    error.hidden = true;
    let body;
    try {
      body = buildDecisionRequest(new FormData(form));
    } catch (problem) {
      showError(problem instanceof Error ? problem.message : "Check the product and pricing inputs.");
      return;
    }
    const submittedInputs = inputsSnapshot();

    submit.disabled = true;
    submit.setAttribute("aria-busy", "true");
    submitLabel.textContent = "Preparing scenario…";
    hasDecision = false;
    showResultMessage("Scoring the candidate ladder…", "Checking the model response and each applicable price rule.");
    try {
      const apiKey = String(new FormData(form).get("api_key") || "");
      const decision = await api.requestDecision(body, apiKey);
      renderDecision(decision, result);
      hasDecision = true;
      staleNotice.hidden = inputsSnapshot() === submittedInputs;
      panel.scrollIntoView({
        behavior: view.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
        block: "nearest",
      });
    } catch (problem) {
      const message = problem instanceof Error ? problem.message : "An unexpected error prevented this decision.";
      showResultMessage("Decision could not be completed.", message);
      showError(message);
    } finally {
      result.hidden = false;
      empty.hidden = true;
      submit.disabled = false;
      submit.removeAttribute("aria-busy");
      submitLabel.textContent = "Get recommendation";
    }
  }

  function markDecisionStale(event) {
    error.hidden = true;
    if (hasDecision && event.target.name !== "api_key") staleNotice.hidden = false;
  }

  form.addEventListener("submit", submitDecision);
  promoToggle.addEventListener("change", () => {
    form.querySelector(".promo-anchor").hidden = !promoToggle.checked;
    promoField.required = promoToggle.checked;
    if (promoToggle.checked) promoField.focus();
  });
  form.addEventListener("input", markDecisionStale);
  form.addEventListener("change", markDecisionStale);
  retryButton.addEventListener("click", refreshServiceState);
  view.addEventListener("online", refreshServiceState);
  documentRoot.addEventListener("visibilitychange", () => {
    if (!documentRoot.hidden) refreshServiceState();
  });

  refreshServiceState();
  return { refreshServiceState };
}
