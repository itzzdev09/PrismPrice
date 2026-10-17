const optionalNumber = (form, name) => {
  const raw = form.elements.namedItem(name).value.trim();
  return raw === "" ? undefined : Number(raw);
};

export function buildRequest(form) {
  const body = {
    sku: form.elements.namedItem("sku").value.trim(),
    as_of: new Date().toISOString(),
    current_price: Number(form.elements.namedItem("current_price").value),
    unit_cost: Number(form.elements.namedItem("unit_cost").value),
    margin_floor_pct: Number(form.elements.namedItem("margin_floor_pct").value) / 100,
    movement_cap_pct: Number(form.elements.namedItem("movement_cap_pct").value) / 100,
    is_promo: form.elements.namedItem("is_promo").checked,
  };
  const competitor = optionalNumber(form, "competitor_price");
  const inventory = optionalNumber(form, "inventory_cover_days");
  if (competitor !== undefined) {
    body.competitor_price = competitor;
    body.competitor_observed_at = new Date().toISOString();
  }
  if (inventory !== undefined) body.inventory_cover_days = inventory;
  if (body.is_promo) {
    body.min_price_last_30d = optionalNumber(form, "min_price_last_30d");
    if (body.min_price_last_30d === undefined) {
      throw new Error("Add the lowest price from the last 30 days to check the promotion fairly.");
    }
  }
  return body;
}

export async function fetchHealth() {
  const response = await fetch("/health", { headers: { Accept: "application/json" } });
  if (!response.ok) throw new Error(`Health check returned ${response.status}.`);
  return response.json();
}

export async function requestDecision(body) {
  const response = await fetch("/decide", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
  });
  const result = await response.json();
  if (!response.ok) {
    const messages = Array.isArray(result.detail)
      ? result.detail.map((item) => item.msg).join(" ")
      : result.detail;
    throw new Error(messages || `Decision request returned ${response.status}.`);
  }
  return result;
}
