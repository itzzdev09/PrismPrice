const errorDetail = (payload) => {
  const detail = payload?.detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => {
      if (!item || typeof item !== "object") return "";
      const field = Array.isArray(item.loc) ? item.loc.slice(1).join(" ") : "";
      const message = typeof item.msg === "string" ? item.msg : "";
      return [field, message].filter(Boolean).join(": ");
    }).filter(Boolean).join(" ").slice(0, 400);
  }
  return typeof detail === "string" ? detail.slice(0, 400) : "";
};

export function createPricingApi({ fetchImpl = globalThis.fetch, timeoutMs = 20_000 } = {}) {
  if (typeof fetchImpl !== "function") throw new TypeError("A fetch implementation is required.");

  async function requestJson(path, options = {}) {
    const controller = new AbortController();
    let timedOut = false;
    const timeout = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, timeoutMs);

    try {
      const response = await fetchImpl(path, {
        ...options,
        cache: "no-store",
        credentials: "omit",
        redirect: "error",
        signal: controller.signal,
      });
      let payload = null;
      try {
        payload = await response.json();
      } catch {
        // Proxies may return an HTML error page or an empty body.
      }

      if (!response.ok) {
        if (response.status === 401) throw new Error("This API requires a valid key. Add it in the API access section and try again.");
        if (response.status === 429) throw new Error("The pricing API is receiving too many requests. Wait a moment and try again.");
        const detail = errorDetail(payload);
        if (detail) throw new Error(detail);
        throw new Error(`The pricing API returned an error (${response.status}). Try again or check the service logs.`);
      }
      if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
        throw new Error("The pricing API returned an unreadable response. Check that the PrismPrice service is running.");
      }
      return payload;
    } catch (problem) {
      if (timedOut) throw new Error("The pricing service took too long to respond. Try again in a moment.");
      if (problem instanceof TypeError) throw new Error("Could not reach the PrismPrice service. Check your connection and try again.");
      throw problem;
    } finally {
      clearTimeout(timeout);
    }
  }

  async function fetchHealth() {
    const health = await requestJson("/health", { headers: { Accept: "application/json" } });
    if (typeof health.model_store_available !== "boolean" || typeof health.policy_version !== "string") {
      throw new Error("The service returned an incomplete health response.");
    }
    return health;
  }

  async function requestDecision(body, apiKey = "") {
    const headers = { "Content-Type": "application/json", Accept: "application/json" };
    if (apiKey.trim()) headers["X-API-Key"] = apiKey.trim();
    return requestJson("/decide", {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });
  }

  return { fetchHealth, requestDecision };
}
