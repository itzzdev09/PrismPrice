const count = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 0 });
const money = new Intl.NumberFormat("en-GB", {
  style: "currency",
  currency: "GBP",
  maximumFractionDigits: 0,
});
const signedPercent = (value) => `${value > 0 ? "+" : ""}${value.toFixed(1)}%`;

function metricCard(label, value, detail) {
  const card = document.createElement("article");
  card.className = "demo-metric-card";
  const title = document.createElement("span");
  title.textContent = label;
  const amount = document.createElement("strong");
  amount.textContent = value;
  const note = document.createElement("small");
  note.textContent = detail;
  card.append(title, amount, note);
  return card;
}

function renderMetrics(data) {
  const { dataset } = data;
  document.querySelector("#decision-form").elements.namedItem("as_of").value =
    `${dataset.decision_as_of}T00:00:00Z`;
  const target = document.querySelector("#demo-metrics");
  target.replaceChildren(
    metricCard("Transaction lines", count.format(dataset.source_rows), "Full public source dataset"),
    metricCard("Distinct SKUs", count.format(dataset.skus), "After source cleaning"),
    metricCard("Markets", count.format(dataset.countries), "Customer countries recorded"),
    metricCard("Net sales", money.format(dataset.net_revenue_gbp), "After recorded returns"),
    metricCard("Modelled products", count.format(dataset.modelable_skus), `${count.format(dataset.model_panel_rows)} observed SKU-days`),
  );

  document.querySelector("#demo-source-summary").textContent =
    `${dataset.name} · ${count.format(dataset.source_rows)} transaction lines · ${dataset.first_date} to ${dataset.last_date} · cleaned to ${count.format(dataset.prepared_rows)} lines`;
}

function renderMonthlyChart(months) {
  const target = document.querySelector("#demo-monthly-chart");
  target.replaceChildren();
  const maximum = Math.max(...months.map((month) => month.revenue));
  const descriptions = [];

  months.forEach((month, index) => {
    const column = document.createElement("div");
    column.className = "demo-month-column";
    column.title = `${month.month}: ${money.format(month.revenue)} net revenue`;

    const track = document.createElement("div");
    track.className = "demo-month-track";
    const bar = document.createElement("span");
    bar.className = "demo-month-bar";
    bar.style.height = `${Math.max(2, (month.revenue / maximum) * 100)}%`;
    track.append(bar);

    const label = document.createElement("span");
    label.className = "demo-month-label";
    if (index % 3 === 0 || index === months.length - 1) {
      const [year, number] = month.month.split("-");
      label.textContent = new Intl.DateTimeFormat("en-GB", {
        month: "short",
        timeZone: "UTC",
      }).format(new Date(Date.UTC(Number(year), Number(number) - 1, 1)));
      if (number === "01" || index === 0) label.textContent += ` '${year.slice(-2)}`;
    }
    column.append(track, label);
    target.append(column);
    descriptions.push(`${month.month}, ${money.format(month.revenue)}`);
  });

  target.setAttribute("aria-label", `Monthly net sales: ${descriptions.join("; ")}`);
}

function renderInsights(data) {
  const { monthly, elasticity, pipeline } = data;
  const completeMonths = monthly.filter((month) => month.month !== "2011-12");
  const peak = completeMonths.reduce((best, month) => month.revenue > best.revenue ? month : best);

  document.querySelector("#demo-season-title").textContent = `${peak.month} was the busiest month`;
  document.querySelector("#demo-season-copy").textContent =
    `${money.format(peak.revenue)} net sales and ${count.format(peak.units)} net units. `
    + `November 2011 was ${signedPercent(pipeline.november_revenue_lift_vs_october_pct)} above October.`;

  document.querySelector("#demo-elasticity-title").textContent =
    `${count.format(elasticity.high_confidence)} of ${count.format(data.dataset.modelable_skus)} SKU estimates passed`;
  document.querySelector("#demo-elasticity-copy").textContent =
    `The remaining ${count.format(elasticity.low_confidence)} use the pooled median elasticity `
    + `${elasticity.median.toFixed(2)}. Estimated range: ${elasticity.p10.toFixed(2)} to ${elasticity.p90.toFixed(2)}. `
    + "Historical price changes were not randomized, so this is not a controlled experiment.";

  document.querySelector("#demo-guardrail-title").textContent =
    `A ${signedPercent(pipeline.aggregate_movement_pct)} catalogue move was held`;
  document.querySelector("#demo-guardrail-copy").textContent =
    `The aggregate movement limit was ±${pipeline.aggregate_movement_limit_pct.toFixed(0)}%. `
    + "Although individual prices passed their checks, a portfolio-level guardrail stopped this saved run from being released.";
}

function describeProduct(product) {
  const base = `${count.format(product.baseline_units)} units per recent active sales day`;
  const method = product.elasticity_confidence === "high"
    ? `SKU estimate ${product.elasticity.toFixed(2)}`
    : `pooled fallback ${product.elasticity.toFixed(2)} (SKU estimate low confidence)`;
  return `${product.observations} historical SKU-days through ${product.last_seen}; ${base}; ${method}. Cost is assumed at a 45% margin.`;
}

function updateWorkbench(product) {
  const form = document.querySelector("#decision-form");
  form.elements.namedItem("sku").value = product.sku;
  form.elements.namedItem("current_price").value = product.current_price.toFixed(2);
  form.elements.namedItem("unit_cost").value = product.unit_cost.toFixed(2);
  document.querySelector("#demo-product-context").textContent = describeProduct(product);
  form.dispatchEvent(new Event("input", { bubbles: true }));
}

function renderProductOptions(products) {
  const select = document.querySelector("#demo-product-select");
  const skuField = document.querySelector('[name="sku"]');
  const options = products.map((product) => {
    const option = new Option(
      `${product.sku} · ${product.description} · ${money.format(product.current_price)}`,
      product.sku,
    );
    return option;
  });
  select.replaceChildren(...options);

  const bySku = new Map(products.map((product) => [product.sku, product]));
  const applySelection = () => {
    const product = bySku.get(select.value);
    if (product) updateWorkbench(product);
  };
  select.addEventListener("change", applySelection);

  const initial = bySku.get(skuField.value) || products[0];
  if (initial) {
    select.value = initial.sku;
    document.querySelector("#demo-product-context").textContent = describeProduct(initial);
  }

  skuField.addEventListener("input", () => {
    const product = bySku.get(skuField.value.trim());
    if (product) {
      select.value = product.sku;
      updateWorkbench(product);
    } else {
      select.value = "";
      document.querySelector("#demo-product-context").textContent =
        "This SKU is not in the historical catalogue; the service will use its generic illustrative curve.";
    }
  });

  document.querySelectorAll("[data-demo-sku]").forEach((button) => {
    button.addEventListener("click", () => {
      select.value = button.dataset.demoSku;
      applySelection();
      document.querySelector("#workbench").scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });
}

function renderTopProducts(products) {
  const target = document.querySelector("#demo-top-products");
  target.replaceChildren();
  products.slice(0, 4).forEach((product, index) => {
    const row = document.createElement("article");
    row.className = "demo-product-row";
    const rank = document.createElement("span");
    rank.className = "demo-product-rank";
    rank.textContent = String(index + 1).padStart(2, "0");
    const details = document.createElement("div");
    details.className = "demo-product-details";
    const title = document.createElement("strong");
    title.textContent = product.description;
    const sku = document.createElement("span");
    sku.textContent = `SKU ${product.sku} · ${count.format(product.observations)} observed days`;
    details.append(title, sku);
    const revenue = document.createElement("span");
    revenue.className = "demo-product-revenue";
    revenue.textContent = money.format(product.observed_revenue);
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.demoSku = product.sku;
    button.textContent = "Try this product";
    row.append(rank, details, revenue, button);
    target.append(row);
  });
}

export function initDemoDataset() {
  fetch("/assets/demo-data.json")
    .then((response) => {
      if (!response.ok) throw new Error("Demo data is unavailable.");
      return response.json();
    })
    .then((data) => {
      renderMetrics(data);
      renderMonthlyChart(data.monthly);
      renderInsights(data);
      renderTopProducts(data.top_products);
      renderProductOptions(data.products);
    })
    .catch(() => {
      document.querySelector("#demo-source-summary").textContent =
        "The historical dataset snapshot could not be loaded.";
      document.querySelector("#demo-product-context").textContent =
        "The data explorer is unavailable; you can still enter a custom SKU and scenario.";
    });
}
