# Online Retail II demo

## Dataset

The browser walkthrough uses **Online Retail II**, a real transaction history
for a UK online giftware retailer covering 1 December 2009 through 9 December
2011. The Kaggle [Online Retail dataset page](https://www.kaggle.com/datasets/lakshmi25npathi/online-retail-dataset)
provides a public mirror. The original dataset is published by the [UCI Machine
Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
under CC BY 4.0. Cite it as: Chen, D. (2012). *Online Retail II*. UCI Machine
Learning Repository. DOI: [10.24432/C5CG6D](https://doi.org/10.24432/C5CG6D).

The repository already contains the UCI source and processed DuckDB store, so
the demo reuses that data rather than checking in another copy of the 43.5 MB
workbook. The browser receives only `web/demo-data.json`: aggregate trends,
catalogue summaries and the saved model review. Customer IDs and transaction
lines are not sent to the browser.

## What visitors can inspect

- 1,067,371 source lines; 1,055,553 lines after the existing data quality
  cleaning; 4,916 distinct SKUs; and 43 customer countries.
- 528,349 daily SKU observations across the cleaned history, totalling about
  11.1 million net units and GBP 19.6 million net sales after recorded returns.
- Monthly sales, several high-revenue modelled products, and the last observed
  price and recent demand baseline for 443 products with at least 300 history
  rows in the saved model run.
- The historical elasticity review: 261 of 443 SKU estimates passed the
  confidence checks. The other 182 were assigned the pooled median estimate
  (about -0.48) for the illustrative scenario.
- A catalogue decision run whose aggregate movement was +9.75%, above its
  +/-3% limit. Individual candidate decisions could pass their SKU guardrails
  while the aggregate release check still held the catalogue.

December 2011 in the monthly chart is partial. The saved decision run is dated
9 December 2011, and its product baselines use the preceding observed history.

## How the interactive example is built

Choosing a product fills in its historical list price and a unit cost calculated
from the run's **assumed 45% gross margin**. At a candidate price, the demo
scales its recent observed baseline using the saved SKU elasticity where it
passed confidence checks, otherwise using the pooled estimate. It derives the
demand range from recent observed variation. This keeps product scale and price
response tied to the dataset, while clearly treating the response curve as a
teaching approximation rather than a deployable trained model.

Cost, inventory and competitor prices are absent from Online Retail II. The
workbench leaves inventory and competitor inputs blank; cost is labelled as an
assumption. Unknown SKUs use the generic illustrative curve. The page labels
the period and model status, and does not present the historical results as
current pricing guidance.

## Rebuild the browser snapshot

After updating the DuckDB data store or saved pipeline review, run from the
repository root:

```powershell
py -3.12 scripts/build_demo_snapshot.py
```

The script writes `src/prismprice/web/demo-data.json`. It requires DuckDB and
the existing files `data/prismprice.duckdb` and `data/runs/real_run.json`.
