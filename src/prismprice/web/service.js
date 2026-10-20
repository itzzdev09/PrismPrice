const optionalNumber = (form, name) => {
  const raw = form.elements.namedItem(name).value.trim();
  if (raw === "") return undefined;
  const value = Number(raw);
  if (!Number.isFinite(value)) throw new Error("Enter a valid number for each optional price or stock field.");
  return value;
};

const requiredNumber = (form, name, label) => {
  const raw = form.elements.namedItem(name).value.trim();
  const value = Number(raw);
  if (raw === "" || !Number.isFinite(value)) throw new Error(`Enter a valid ${label}.`);
  return value;
};

export function buildRequest(form) {
  const sku = form.elements.namedItem("sku").value.trim();
  if (!sku) throw new Error("Enter a product SKU before requesting a price.");
  if (sku.length > 128) throw new Error("Product SKUs can contain up to 128 characters.");

  const currentPrice = requiredNumber(form, "current_price", "current price");
  const unitCost = requiredNumber(form, "unit_cost", "unit cost");
  const marginFloor = requiredNumber(form, "margin_floor_pct", "margin floor");
  const movementCap = requiredNumber(form, "movement_cap_pct", "price move cap");
  if (currentPrice <= 0 || unitCost <= 0) throw new Error("Current price and unit cost must be greater than zero.");
  if (marginFloor < 0 || marginFloor > 100) throw new Error("Margin floor must be between 0% and 100%.");
  if (movementCap < 0 || movementCap > 50) throw new Error("Price move cap must be between 0% and 50%.");

  const asOf = new Date().toISOString();
  const body = {
    sku,
    as_of: asOf,
    current_price: currentPrice,
    unit_cost: unitCost,
    margin_floor_pct: marginFloor / 100,
    movement_cap_pct: movementCap / 100,
    is_promo: form.elements.namedItem("is_promo").checked,
  };
  const competitor = optionalNumber(form, "competitor_price");
  const inventory = optionalNumber(form, "inventory_cover_days");
  if (competitor !== undefined) {
    if (competitor <= 0) throw new Error("Competitor price must be greater than zero.");
    body.competitor_price = competitor;
    body.competitor_observed_at = asOf;
  }
  if (inventory !== undefined) {
    if (inventory < 0) throw new Error("Stock cover cannot be negative.");
    body.inventory_cover_days = inventory;
  }
  if (body.is_promo) {
    body.min_price_last_30d = optionalNumber(form, "min_price_last_30d");
    if (body.min_price_last_30d === undefined) throw new Error("Add the lowest price from the last 30 days to check the promotion fairly.");
    if (body.min_price_last_30d <= 0) throw new Error("The 30-day reference price must be greater than zero.");
  }
  return body;
}

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

async function requestJson(path, options = {}) {
  const controller = new AbortController();
  let timedOut = false;
  const timeout = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, 20000);

  try {
    const response = await fetch(path, {
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
      // Error pages from a proxy or server may be HTML or empty.
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
    window.clearTimeout(timeout);
  }
}

export async function fetchHealth() {
  const health = await requestJson("/health", { headers: { Accept: "application/json" } });
  if (typeof health.model_store_available !== "boolean" || typeof health.policy_version !== "string") {
    throw new Error("The service returned an incomplete health response.");
  }
  return health;
}

export async function requestDecision(body, apiKey = "") {
  const headers = { "Content-Type": "application/json", Accept: "application/json" };
  if (apiKey.trim()) headers["X-API-Key"] = apiKey.trim();
  return requestJson("/decide", {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
}
