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

function renderMonthlyChart(months, metric = "revenue") {
  const target = document.querySelector("#demo-monthly-chart");
  target.replaceChildren();
  const valueFor = (month) => month[metric];
  const formatValue = metric === "revenue"
    ? (value) => money.format(value)
    : (value) => `${count.format(value)} net units`;
  const title = document.querySelector("#demo-monthly-title");
  title.textContent = metric === "revenue" ? "Monthly net revenue" : "Monthly net units";
  const maximum = Math.max(...months.map(valueFor));
  const descriptions = [];

  months.forEach((month, index) => {
    const column = document.createElement("div");
    column.className = "demo-month-column";
    column.title = `${month.month}: ${formatValue(valueFor(month))}`;

    const track = document.createElement("div");
    track.className = "demo-month-track";
    const bar = document.createElement("span");
    bar.className = "demo-month-bar";
    bar.style.height = `${Math.max(2, (valueFor(month) / maximum) * 100)}%`;
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
    descriptions.push(`${month.month}, ${formatValue(valueFor(month))}`);
  });

  target.setAttribute("aria-label", `Monthly ${metric === "revenue" ? "net revenue" : "net units"}: ${descriptions.join("; ")}`);
}

const svgNamespace = "http://www.w3.org/2000/svg";
function svgNode(name, attributes = {}, text) {
  const node = document.createElementNS(svgNamespace, name);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderSeasonalityChart(monthly) {
  const target = document.querySelector("#demo-seasonality-chart");
  target.replaceChildren();
  const completeMonths = monthly.filter((month) => month.month !== "2011-12");
  const monthNames = new Intl.DateTimeFormat("en-GB", { month: "short", timeZone: "UTC" });
  const grouped = new Map();
  completeMonths.forEach((row) => {
    const monthNumber = Number(row.month.slice(5, 7));
    const values = grouped.get(monthNumber) || [];
    values.push(row.revenue);
    grouped.set(monthNumber, values);
  });
  const averageMonth = completeMonths.reduce((sum, row) => sum + row.revenue, 0) / completeMonths.length;
  const seasonality = [...grouped.entries()].map(([month, values]) => ({
    month,
    revenue: values.reduce((sum, value) => sum + value, 0) / values.length,
    index: (values.reduce((sum, value) => sum + value, 0) / values.length / averageMonth) * 100,
  }));
  const maximum = Math.max(150, Math.ceil(Math.max(...seasonality.map((item) => item.index)) / 25) * 25);
  const svg = svgNode("svg", { viewBox: "0 0 660 245", role: "img", "aria-label": "Average monthly revenue by calendar month, indexed to the overall average complete month" });
  svg.append(svgNode("title", {}, "Seasonality index by calendar month"));
  svg.append(svgNode("desc", {}, "Bars compare average net revenue for each calendar month with the average across all complete months. The horizontal line marks 100 percent."));
  const left = 42;
  const right = 650;
  const top = 18;
  const bottom = 202;
  const y = (value) => bottom - (value / maximum) * (bottom - top);
  const ticks = [0, 50, 100, 150, maximum].filter((value, index, values) => values.indexOf(value) === index && value <= maximum);
  ticks.forEach((tick) => {
    const line = svgNode("line", { x1: left, x2: right, y1: y(tick), y2: y(tick), class: tick === 100 ? "seasonality-baseline" : "chart-gridline" });
    svg.append(line, svgNode("text", { x: left - 8, y: y(tick) + 3, class: "chart-axis-label", "text-anchor": "end" }, `${tick}%`));
  });
  const slot = (right - left) / seasonality.length;
  seasonality.forEach((item, index) => {
    const center = left + slot * (index + 0.5);
    const barTop = y(item.index);
    const monthName = monthNames.format(new Date(Date.UTC(2020, item.month - 1, 1)));
    const rect = svgNode("rect", {
      x: center - Math.min(17, slot * 0.32),
      y: barTop,
      width: Math.min(34, slot * 0.64),
      height: bottom - barTop,
      rx: 4,
      class: item.index >= 100 ? "seasonality-bar above" : "seasonality-bar",
    });
    rect.append(svgNode("title", {}, `${monthName}: ${Math.round(item.index)}% of the average month · ${money.format(item.revenue)} average net revenue`));
    svg.append(rect);
    svg.append(svgNode("text", { x: center, y: bottom + 18, class: "chart-axis-label chart-month-label", "text-anchor": "middle" }, monthName));
  });
  target.append(svg);
  const description = seasonality.map((item) => {
    const monthName = monthNames.format(new Date(Date.UTC(2020, item.month - 1, 1)));
    return `${monthName}, ${Math.round(item.index)} percent of average monthly revenue`;
  });
  target.setAttribute("aria-label", `Seasonality index: ${description.join("; ")}`);
}

function renderElasticityChart(elasticity) {
  const target = document.querySelector("#demo-elasticity-chart");
  target.replaceChildren();
  const priceMoves = [-10, -5, 0, 5, 10];
  const elasticities = [elasticity.p10, elasticity.median, elasticity.p90];
  const response = priceMoves.map((move) => elasticities.map((estimate) => ((1 + move / 100) ** estimate - 1) * 100));
  const values = response.flat();
  const min = Math.min(0, ...values);
  const max = Math.max(0, ...values);
  const padding = Math.max(1, (max - min) * 0.12);
  const yMin = min - padding;
  const yMax = max + padding;
  const svg = svgNode("svg", { viewBox: "0 0 660 250", role: "img", "aria-label": "Modeled change in demand across price changes from minus ten to plus ten percent" });
  svg.append(svgNode("title", {}, "Modeled demand response by price move"));
  svg.append(svgNode("desc", {}, `At each price move, the shaded band spans P10 to P90 elasticity estimates. The solid line uses median elasticity ${elasticity.median.toFixed(2)}.`));
  const left = 52;
  const right = 646;
  const top = 14;
  const bottom = 204;
  const x = (move) => left + ((move + 10) / 20) * (right - left);
  const y = (value) => bottom - ((value - yMin) / (yMax - yMin)) * (bottom - top);
  const tickCount = 4;
  for (let index = 0; index <= tickCount; index += 1) {
    const value = yMin + ((yMax - yMin) * index) / tickCount;
    const yPosition = y(value);
    svg.append(svgNode("line", { x1: left, x2: right, y1: yPosition, y2: yPosition, class: Math.abs(value) < 0.0001 ? "chart-zero-line" : "chart-gridline" }));
    const label = `${value > 0 ? "+" : ""}${value.toFixed(0)}%`;
    svg.append(svgNode("text", { x: left - 9, y: yPosition + 3, class: "chart-axis-label", "text-anchor": "end" }, label));
  }
  const lower = response.map((point) => Math.min(point[0], point[2]));
  const upper = response.map((point) => Math.max(point[0], point[2]));
  const bandPoints = [
    ...upper.map((value, index) => `${x(priceMoves[index])},${y(value)}`),
    ...lower.map((value, index) => `${x(priceMoves[index])},${y(value)}`).reverse(),
  ];
  svg.append(svgNode("polygon", { points: bandPoints.join(" "), class: "elasticity-range" }));
  const drawLine = (seriesIndex, className) => {
    const points = response.map((point, index) => `${index ? "L" : "M"}${x(priceMoves[index])},${y(point[seriesIndex])}`).join(" ");
    svg.append(svgNode("path", { d: points, class: className }));
  };
  drawLine(0, "elasticity-boundary");
  drawLine(2, "elasticity-boundary");
  drawLine(1, "elasticity-median");
  priceMoves.forEach((move) => {
    svg.append(svgNode("text", { x: x(move), y: bottom + 19, class: "chart-axis-label", "text-anchor": "middle" }, `${move > 0 ? "+" : ""}${move}%`));
  });
  svg.append(svgNode("text", { x: (left + right) / 2, y: 246, class: "chart-axis-title", "text-anchor": "middle" }, "Price change"));
  target.append(svg);
  target.setAttribute("aria-label", `Modeled demand response: ${priceMoves.map((move, index) => {
    const range = [response[index][0], response[index][2]].sort((a, b) => a - b);
    return `${move > 0 ? "plus " : ""}${move} percent price move, estimated demand change ${range[0].toFixed(1)} to ${range[1].toFixed(1)} percent, median ${response[index][1].toFixed(1)} percent`;
  }).join("; ")}`);
}

function renderInsights(data) {
  const { monthly, elasticity, pipeline } = data;
  const completeMonths = monthly.filter((month) => month.month !== "2011-12");
  const monthNames = new Intl.DateTimeFormat("en-GB", { month: "long", timeZone: "UTC" });
  const byMonth = new Map();
  completeMonths.forEach((row) => {
    const monthNumber = Number(row.month.slice(5, 7));
    const rows = byMonth.get(monthNumber) || [];
    rows.push(row.revenue);
    byMonth.set(monthNumber, rows);
  });
  const seasonalPeak = [...byMonth.entries()].map(([month, values]) => ({
    month,
    average: values.reduce((sum, value) => sum + value, 0) / values.length,
  })).reduce((best, month) => month.average > best.average ? month : best);
  const priorYearRevenue = completeMonths.filter((month) => month.month.startsWith("2010-") && month.month.slice(5, 7) !== "12").reduce((sum, month) => sum + month.revenue, 0);
  const currentYearRevenue = completeMonths.filter((month) => month.month.startsWith("2011-") && month.month.slice(5, 7) !== "12").reduce((sum, month) => sum + month.revenue, 0);
  const matchedMonthChange = priorYearRevenue ? (currentYearRevenue / priorYearRevenue - 1) * 100 : 0;
  const fivePercentDemandChange = ((1.05 ** elasticity.median) - 1) * 100;
  const demandRange = [((1.05 ** elasticity.p10) - 1) * 100, ((1.05 ** elasticity.p90) - 1) * 100].sort((a, b) => a - b);

  document.querySelector("#demo-season-title").textContent = `${signedPercent(matchedMonthChange)} net revenue, Jan–Nov year over year`;
  document.querySelector("#demo-season-copy").textContent =
    `The average ${monthNames.format(new Date(Date.UTC(2020, seasonalPeak.month - 1, 1)))} is the strongest calendar-month pattern. `
    + `November 2011 was ${signedPercent(pipeline.november_revenue_lift_vs_october_pct)} above October; these comparisons do not isolate pricing effects.`;

  document.querySelector("#demo-elasticity-title").textContent = `A +5% price move maps to ${signedPercent(fivePercentDemandChange)} median demand`;
  document.querySelector("#demo-elasticity-copy").textContent =
    `P10–P90 elasticity estimates imply a ${signedPercent(demandRange[0])} to ${signedPercent(demandRange[1])} demand range. `
    + `${count.format(elasticity.high_confidence)} of ${count.format(data.dataset.modelable_skus)} SKU estimates were high confidence; this is a model scenario, not an observed outcome.`;

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
      renderMonthlyChart(data.monthly, "revenue");
      document.querySelectorAll("[data-monthly-metric]").forEach((button) => {
        button.addEventListener("click", () => {
          const metric = button.dataset.monthlyMetric;
          document.querySelectorAll("[data-monthly-metric]").forEach((option) => {
            option.setAttribute("aria-pressed", String(option === button));
          });
          renderMonthlyChart(data.monthly, metric);
        });
      });
      renderSeasonalityChart(data.monthly);
      renderElasticityChart(data.elasticity);
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
