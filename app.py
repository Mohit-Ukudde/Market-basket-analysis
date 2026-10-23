from __future__ import annotations

import io
from typing import Iterable

import pandas as pd
import plotly.express as px
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules


st.set_page_config(page_title="Market Basket Analysis", page_icon="🛒", layout="wide")


SAMPLE_TRANSACTIONS = [
    ["Bread", "Milk", "Eggs"],
    ["Bread", "Butter", "Jam"],
    ["Milk", "Eggs", "Cereal"],
    ["Bread", "Milk", "Butter"],
    ["Coffee", "Milk", "Sugar"],
    ["Bread", "Eggs", "Cheese"],
    ["Milk", "Cereal", "Bananas"],
    ["Bread", "Butter", "Eggs", "Cheese"],
    ["Coffee", "Milk", "Cookies"],
    ["Bread", "Milk", "Eggs", "Butter"],
    ["Cereal", "Bananas", "Yogurt"],
    ["Bread", "Jam", "Peanut Butter"],
    ["Milk", "Eggs", "Cheese"],
    ["Coffee", "Cookies", "Sugar"],
    ["Bread", "Milk", "Cheese"],
    ["Bananas", "Yogurt", "Granola"],
    ["Bread", "Butter", "Jam", "Peanut Butter"],
    ["Milk", "Cereal", "Yogurt"],
    ["Coffee", "Milk", "Cookies", "Sugar"],
    ["Bread", "Eggs", "Butter"],
]


def transactions_to_matrix(transactions: Iterable[Iterable[str]]) -> pd.DataFrame:
    rows = [set(str(item).strip() for item in basket if str(item).strip()) for basket in transactions]
    items = sorted({item for basket in rows for item in basket})
    return pd.DataFrame([{item: item in basket for item in items} for basket in rows])


def sample_matrix() -> pd.DataFrame:
    return transactions_to_matrix(SAMPLE_TRANSACTIONS)


def parse_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Accept either a basket-per-row CSV or a long transaction/item CSV."""
    raw = pd.read_csv(uploaded_file)
    normalized = {str(column).strip().lower(): column for column in raw.columns}
    transaction_column = next((normalized[k] for k in ("transaction", "transaction_id", "invoice", "basket") if k in normalized), None)
    item_column = next((normalized[k] for k in ("item", "product", "description") if k in normalized), None)
    if transaction_column and item_column:
        grouped = raw.dropna(subset=[transaction_column, item_column]).groupby(transaction_column)[item_column]
        return transactions_to_matrix(grouped.apply(list).tolist())

    # Treat each row as one basket. Empty cells are ignored.
    return transactions_to_matrix(raw.fillna("").astype(str).values.tolist())


def itemset_label(itemset: frozenset | set) -> str:
    return " + ".join(sorted(itemset))


def run_analysis(matrix: pd.DataFrame, min_support: float, min_confidence: float):
    frequent = apriori(matrix.astype(bool), min_support=min_support, use_colnames=True)
    if frequent.empty:
        return frequent, pd.DataFrame()
    frequent["itemset_label"] = frequent["itemsets"].map(itemset_label)
    rules = association_rules(frequent, metric="confidence", min_threshold=min_confidence)
    if not rules.empty:
        rules["antecedents_label"] = rules["antecedents"].map(itemset_label)
        rules["consequents_label"] = rules["consequents"].map(itemset_label)
        rules["rule_label"] = rules["antecedents_label"] + " → " + rules["consequents_label"]
        rules = rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)
    return frequent.sort_values("support", ascending=False).reset_index(drop=True), rules


st.title("🛒 Market Basket Analysis")
st.caption("Discover product associations that can improve cross-selling, bundles, and store layout decisions.")

with st.sidebar:
    st.header("Analysis controls")
    uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])
    st.caption("Supported formats: one basket per row, or long format with transaction + item columns.")
    min_support = st.slider("Minimum support", 0.05, 0.80, 0.15, 0.01)
    min_confidence = st.slider("Minimum confidence", 0.10, 1.00, 0.50, 0.01)
    top_n = st.slider("Rows to display", 5, 25, 10)

try:
    matrix = parse_uploaded_file(uploaded) if uploaded else sample_matrix()
except Exception as exc:
    st.error(f"Could not read the uploaded file: {exc}")
    st.stop()

if matrix.empty or matrix.shape[1] == 0:
    st.warning("No products were found. Add at least one non-empty transaction row.")
    st.stop()

frequent, rules = run_analysis(matrix, min_support, min_confidence)
transactions = len(matrix)
total_items = matrix.shape[1]
avg_basket = matrix.sum(axis=1).mean()

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Transactions", f"{transactions:,}")
metric2.metric("Unique products", f"{total_items:,}")
metric3.metric("Avg. basket size", f"{avg_basket:.1f}")
metric4.metric("Rules found", f"{len(rules):,}")

if frequent.empty:
    st.warning("No frequent itemsets meet the selected support. Lower minimum support and try again.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Frequent itemsets", "Association rules", "Business insights"])

with tab1:
    st.subheader("Products that appear together frequently")
    itemsets_display = frequent.head(top_n)[["itemset_label", "support"]].copy()
    itemsets_display["support"] = (itemsets_display["support"] * 100).round(1)
    itemsets_display = itemsets_display.rename(columns={"itemset_label": "Itemset", "support": "Support (%)"})
    st.dataframe(itemsets_display, use_container_width=True, hide_index=True)

    chart_data = frequent.head(top_n).sort_values("support")
    fig = px.bar(chart_data, x="support", y="itemset_label", orientation="h", text_auto=".1%",
                 labels={"support": "Support", "itemset_label": "Itemset"},
                 title="Top frequent itemsets by support", color="support", color_continuous_scale="Blues")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Association rules for cross-selling")
    if rules.empty:
        st.info("No rules meet the selected confidence threshold. Lower minimum confidence or support.")
    else:
        display = rules.head(top_n)[["rule_label", "support", "confidence", "lift"]].copy()
        display["support"] = (display["support"] * 100).round(1).astype(str) + "%"
        display["confidence"] = (display["confidence"] * 100).round(1).astype(str) + "%"
        display["lift"] = display["lift"].round(2)
        display.columns = ["Rule", "Support", "Confidence", "Lift"]
        st.dataframe(display, use_container_width=True, hide_index=True)

        plot_rules = rules.head(top_n).sort_values("lift")
        fig = px.scatter(plot_rules, x="confidence", y="lift", size="support", color="lift",
                         hover_name="rule_label", labels={"confidence": "Confidence", "lift": "Lift"},
                         title="Rule strength: confidence vs. lift", color_continuous_scale="Viridis")
        fig.update_xaxes(tickformat=".0%")
        fig.add_hline(y=1, line_dash="dash", line_color="gray", annotation_text="Lift = 1 (no association)")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Recommended actions")
    if rules.empty:
        st.info("Generate rules first to receive recommendations.")
    else:
        strong_rules = rules[(rules["lift"] > 1) & (rules["confidence"] >= min_confidence)].head(5)
        for _, rule in strong_rules.iterrows():
            st.markdown(
                f"**Promote {rule['consequents_label']} when customers choose {rule['antecedents_label']}** — "
                f"{rule['confidence']:.0%} confidence, {rule['lift']:.2f}× lift, and {rule['support']:.1%} support."
            )
        st.markdown("**How to use these findings**")
        st.markdown("- Place high-lift pairs together or show them as checkout recommendations.\n"
                    "- Create bundles for rules with both strong support and confidence.\n"
                    "- Use the highest-support itemsets for inventory planning and promotions.\n"
                    "- Treat lift above 1 as evidence of a positive association, not proof of causation.")

with st.expander("Preview transaction matrix"):
    st.dataframe(matrix.astype(int), use_container_width=True, height=220)

