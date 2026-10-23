# Market Basket Analysis

An interactive Streamlit dashboard that applies the Apriori algorithm to discover product associations and support cross-selling decisions.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The app opens with a built-in grocery dataset. Use the sidebar to upload a CSV and tune minimum support and confidence.

## Input formats

The uploader accepts either:

1. One transaction per row, with each product in its own column.
2. Long format with columns named `transaction` (or `transaction_id`, `invoice`, `basket`) and `item` (or `product`, `description`).

## Methodology

- Support: how often an itemset appears across all transactions.
- Confidence: how often the consequent appears when the antecedent appears.
- Lift: association strength relative to random co-occurrence; lift above 1 indicates a positive association.

The dashboard includes frequent-itemset rankings, an itemset bar chart, a confidence/lift scatter plot, and rule-based recommendations.
