# Historical demand workbench

## Dataset

The workbench uses **Online Retail II**, a transaction history for a UK online
giftware retailer covering 1 December 2009 through 9 December 2011. The public
source is available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
under CC BY 4.0 and from a [Kaggle mirror](https://www.kaggle.com/datasets/lakshmi25npathi/online-retail-dataset).
Cite it as Chen, D. (2012), *Online Retail II*, UCI Machine Learning
Repository, DOI [10.24432/C5CG6D](https://doi.org/10.24432/C5CG6D).

The repository contains the prepared DuckDB store, so the model bundle can be
built without downloading another copy of the workbook. The browser receives
aggregate charts and a small product catalogue; customer IDs and transaction
lines are not sent to it.

## Connected request path

The local workbench now connects the following components for one historical
SKU request:

1. `FeatureBuilder` creates features strictly before the requested `as_of`.
2. The service loads a digest-checked quantile model and verifies its data
   snapshot and feature schema.
3. The model predicts P10/P50/P90 demand for every price in the candidate
   ladder. Demand is measured as positive units per observed active sales day.
4. The decision engine scores expected contribution and downside risk, then
   applies hard guardrails.
5. The service stores the model, data, feature, candidate, and guardrail
   evidence in the local DuckDB decision ledger before returning the result.

The result is a reviewable historical scenario. It is not published to a
retailer.

## Building the model bundle

Install the focused local-workbench dependencies and build the bundle from the
repository root. The UCI workbook is not included; download and extract
`online_retail_II.xlsx` from the [UCI dataset page](https://archive.ics.uci.edu/dataset/502/online+retail+ii),
then create the local transaction store and model bundle:

```powershell
py -3.12 -m pip install -e ".[demo]"
py -3.12 scripts/ingest_uci.py path\to\online_retail_II.xlsx
py -3.12 scripts/build_demand_bundle.py
```

The script reads the prepared daily panel from `data/prismprice.duckdb`, builds
point-in-time training frames for SKUs with at least 300 active sales days, and
compares LightGBM with scikit-learn's histogram gradient boosting quantile
model on three expanding temporal folds. It records WAPE, pinball loss, P10
tail coverage, 80% interval coverage, and per-fold metrics. A candidate must
pass both 5 percentage point interval and tail calibration gates. If neither
candidate passes, the script leaves the comparison report for review and does
not publish a serving bundle.

Among candidates that pass both gates, the lowest pooled pinball-loss model is
the primary and the next best is stored as the standby. The standby is a
separate fitted artifact and is used only if the primary fails while
forecasting. If only one candidate passes calibration, the bundle records that
no standby is available and the service retains its current-price hold for a
primary prediction failure.

The output in `artifacts/demand-bundle/` contains the primary and optional
standby model files plus a manifest with each artifact's SHA-256 digest,
source-panel snapshot, supported SKUs, temporal validation, and assumptions.
At startup, a valid standby is promoted if the primary artifact fails integrity
or loading checks. If the standby itself is invalid, the primary remains
available and the problem is exposed in health context. `artifacts/` is
gitignored. Rebuild after changing the source panel. The app verifies each
artifact's digest and feature schema and checks both models against the active
source-panel snapshot.

## Interpretation and limits

- Source rows are positive sales on active SKU-days. A missing date is not
  imputed as zero because the source has no item-availability or stockout flag.
- The model estimates historical demand associations. Prices were not
  randomized, so the model does not identify the causal effect of changing a
  price.
- The forecast unit is **units per active sales day**, not units per calendar
  day, week, or month.
- Costs, inventory, competitor prices, customer retention, and category
  substitution are absent from the source. The form accepts a cost assumption
  for contribution scoring and guardrails; other unavailable layers are shown
  explicitly.
- CLV, causal elasticity, portfolio optimization, experiments, production
  monitoring, retailer feeds, approval workflows, and price publishing are
  separate research or future deployment work. They do not feed this request
  path.
- An unsupported SKU or date causes the service to hold the supplied current
  price and record the reason. A primary prediction failure uses the standby
  only when it passed the same temporal calibration gates; if both models fail,
  the service holds the supplied current price. There is no generic fallback
  demand curve.

The dataset page also contains offline historical elasticity and portfolio
analyses. Those are clearly labeled as separate research scenarios and are not
the model used by the workbench.

## Rebuild the browser snapshot

After updating the DuckDB store or saved historical analysis, run:

```powershell
py -3.12 scripts/build_demo_snapshot.py
```

The script writes `src/prismprice/web/demo-data.json`. It reads
`data/prismprice.duckdb` and `data/runs/real_run.json`.
