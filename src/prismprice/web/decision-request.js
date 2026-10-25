const rawValue = (formData, name) => String(formData.get(name) ?? "").trim();

const optionalNumber = (formData, name) => {
  const raw = rawValue(formData, name);
  if (raw === "") return undefined;
  const value = Number(raw);
  if (!Number.isFinite(value)) throw new Error("Enter a valid number for each optional price or stock field.");
  return value;
};

const requiredNumber = (formData, name, label) => {
  const raw = rawValue(formData, name);
  const value = Number(raw);
  if (raw === "" || !Number.isFinite(value)) throw new Error(`Enter a valid ${label}.`);
  return value;
};

export function buildDecisionRequest(formData, now = () => new Date().toISOString()) {
  const sku = rawValue(formData, "sku");
  if (!sku) throw new Error("Enter a product SKU before requesting a price.");
  if (sku.length > 128) throw new Error("Product SKUs can contain up to 128 characters.");

  const currentPrice = requiredNumber(formData, "current_price", "current price");
  const unitCost = requiredNumber(formData, "unit_cost", "unit cost");
  const marginFloor = requiredNumber(formData, "margin_floor_pct", "margin floor");
  const movementCap = requiredNumber(formData, "movement_cap_pct", "price move cap");
  if (currentPrice <= 0 || unitCost <= 0) throw new Error("Current price and unit cost must be greater than zero.");
  if (marginFloor < 0 || marginFloor > 100) throw new Error("Margin floor must be between 0% and 100%.");
  if (movementCap < 0 || movementCap > 50) throw new Error("Price move cap must be between 0% and 50%.");

  const asOf = rawValue(formData, "as_of") || now();
  const body = {
    sku,
    as_of: asOf,
    current_price: currentPrice,
    unit_cost: unitCost,
    margin_floor_pct: marginFloor / 100,
    movement_cap_pct: movementCap / 100,
    is_promo: formData.get("is_promo") === "on",
  };

  const competitor = optionalNumber(formData, "competitor_price");
  const inventory = optionalNumber(formData, "inventory_cover_days");
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
    body.min_price_last_30d = optionalNumber(formData, "min_price_last_30d");
    if (body.min_price_last_30d === undefined) throw new Error("Add the lowest price from the last 30 days to check the promotion fairly.");
    if (body.min_price_last_30d <= 0) throw new Error("The 30-day reference price must be greater than zero.");
  }
  return body;
}
