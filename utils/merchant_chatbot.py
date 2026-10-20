
"""Schema-aware merchant analytics with deterministic Pandas execution.

The router infers metrics, grouping columns, filters, and rankings from the
question and the uploaded dataframe. It does not execute LLM-generated code.
"""
import json
import re
import pandas as pd


def _normalise(value):
    return re.sub(r"\s+", " ", str(value).replace("_", " ").strip().lower())


def _column_map(df):
    return {_normalise(c): c for c in df.columns}


def _resolve_column(df, phrase):
    phrase = _normalise(phrase)
    cmap = _column_map(df)
    if phrase in cmap:
        return cmap[phrase]
    aliases = {
        "merchant": ["merchant name", "business", "store"],
        "merchant name": ["merchant"],
        "payment method": ["payment type", "mode of payment"],
        "country": ["region"],
        "amount": ["transaction amount", "transaction value", "sales", "revenue", "value"],
        "status": ["transaction status", "payment status"],
        "currency": ["currencies"],
        "transaction id": ["transaction ids", "id", "records"],
        "category": ["product category"],
    }
    for canonical, variants in aliases.items():
        if phrase == canonical or phrase in variants:
            for candidate in [canonical, *variants]:
                if candidate in cmap:
                    return cmap[candidate]
    return None


def _mentioned_column(df, text):
    """Find an actual dataframe column mentioned in a natural-language question."""
    q = _normalise(text)
    columns = sorted(df.columns, key=lambda c: len(_normalise(c)), reverse=True)
    for column in columns:
        name = _normalise(column)
        if name and re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", q):
            return column
    # Resolve common business language against columns that actually exist.
    for alias in (
        "merchant", "payment method", "country", "amount", "status",
        "currency", "category", "transaction id",
    ):
        if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", q):
            resolved = _resolve_column(df, alias)
            if resolved is not None:
                return resolved
    return None


def _flag_mask(series, label):
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip().str.casefold()
    truthy = numeric.eq(1) | text.isin(["true", "yes", "y", "1", label, label + "ed"])
    if label == "refund":
        truthy = truthy | text.isin(["refunded", "refund"])
    return truthy.fillna(False)


def _status_mask(df, status_col, target):
    return df[status_col].astype("string").str.strip().str.casefold().eq(target.casefold()).fillna(False)


def _safe_records(df, limit=30):
    return df.head(limit).astype(object).where(pd.notna(df.head(limit)), None).to_dict("records")


def _out(operation, result, explanation=""):
    return {"supported": True, "operation": operation, "result": result, "explanation": explanation}


def _unsupported(message=None):
    return {
        "supported": False,
        "operation": "unsupported_question",
        "result": message or (
            "I couldn't map the question to a safe analysis. Try requesting a count, "
            "sum, average, rate, ranking, or breakdown using columns in the uploaded data."
        ),
    }


def analyze_merchant_question(dataframe, question):
    """Parse a question into a supported analytical operation and calculate it."""

def answer_merchant_question(dataframe, question, llm=None):
    """Format deterministic results from the existing Pandas analytics engine."""
    analysis = analyze_merchant_question(dataframe, question)

    if not analysis.get("supported"):
        return analysis.get("result", "I could not analyze that question.")

    op = analysis["operation"]
    result = analysis["result"]

    if op == "row_count":
        return f"The dataset contains {result['count']:,} rows/transactions."

    if op == "column_overview":
        columns = "\n".join(f"- {c}" for c in result["columns"])
        return (
            f"The dataset has {result['row_count']:,} rows and "
            f"{result['column_count']} columns:\n{columns}"
        )

    if op == "data_quality":
        return (
            f"Missing values: {result['missing_value_count']:,}\n"
            f"Duplicate rows/IDs: {result['duplicate_rows']}"
        )

    if op == "dataset_overview":
        return (
            f"Rows: {result['row_count']:,}\n"
            f"Columns: {result['column_count']}\n"
            f"Missing values: {result['missing_value_count']:,}\n"
            f"Duplicate rows: {result['duplicate_rows']}"
        )

    if op == "status_count":
        return (
            f"There are {result['count']:,} transactions with status "
            f"'{result['status']}'."
        )

    if op == "status_records":
        import json
        return (
            f"Found {result['count']:,} transactions with status "
            f"'{result['status']}':\n"
            + json.dumps(result["records"], ensure_ascii=False, indent=2, default=str)
        )

    if op == "outcome_count":
        labels = {
            "refund": "refunds",
            "chargeback": "chargebacks",
            "failed": "failed transactions",
        }
        return f"Found {result['count']:,} {labels.get(result['outcome'], result['outcome'])}."

    if op == "overall_rate":
        return (
            f"{result['outcome'].title()} rate: {result['rate_percent']:.2f}% "
            f"({result['count']}/{result['denominator']} transactions)."
        )

    if op in ("grouped_rate", "grouped_count", "grouped_sum"):
        rows = result.get("rows", [])
        if not rows:
            return "No matching groups were found."

        group_columns = result.get("group_columns") or [result.get("group_column")]
        lines = []

        for row in rows:
            labels = [
                f"{column}={row.get(column)}"
                for column in group_columns
                if column
            ]
            label = ", ".join(labels)

            if "total_amount" in row:
                line = (
                    f"- {label}: total {row['total_amount']:,.2f} "
                    f"({row.get('transaction_count', 0)} transactions)"
                )
            elif "rate_percent" in row:
                line = (
                    f"- {label}: {row['rate_percent']:.2f}% rate "
                    f"({row.get('outcome_count', 0)}/"
                    f"{row.get('transaction_count', 0)} transactions; "
                    f"outcome={result.get('outcome')})"
                )
            else:
                line = (
                    f"- {label}: "
                    f"{row.get('outcome_count', row.get('transaction_count', 0))} "
                    f"transactions"
                )

            lines.append(line)

        heading = "Analysis results:"
        if result.get("outcome") == "chargeback":
            heading = "Chargeback counts by group:"
        elif result.get("outcome") == "refund":
            heading = "Refund results by group:"
        elif result.get("outcome") == "failed":
            heading = "Failure results by group:"

        return heading + "\n" + "\n".join(lines)

    if op == "merchant_amount":
        key = "average_amount" if "average_amount" in result else "total_amount"
        value = result.get(key)

        if value is None:
            return f"No valid numeric amounts found for {result['merchant']}."

        currencies = result.get("currencies", [])
        currency_note = f" ({', '.join(currencies)})" if currencies else ""

        return (
            f"{key.replace('_', ' ').title()} for {result['merchant']}: "
            f"{value:,.2f}{currency_note} across "
            f"{result['transaction_count']} transactions."
        )

    if op in ("numeric_sum", "numeric_mean", "numeric_median"):
        key = {
            "numeric_sum": "sum",
            "numeric_mean": "mean",
            "numeric_median": "median",
        }[op]
        return (
            f"{key.title()} of '{result['column']}': "
            f"{result[key]:,.3f} from {result['count']} valid numeric values."
        )

    if op == "outcome_records":
        import json
        return (
            f"Found {result['count']} matching records:\n"
            + json.dumps(result["records"], ensure_ascii=False, indent=2, default=str)
        )

    import json
    return json.dumps(result, ensure_ascii=False, indent=2, default=str)

def analyze_merchant_question(dataframe, question):
    """Calculate spreadsheet answers deterministically with pandas."""
    import pandas as pd

    if dataframe is None or dataframe.empty:
        return {"supported": False, "operation": "no_data",
                "result": "Load a non-empty spreadsheet first."}

    q = _normalise(question)
    cols = dataframe.columns.tolist()
    merchant = _resolve_column(dataframe, "merchant")
    amount = _resolve_column(dataframe, "amount")
    currency = _resolve_column(dataframe, "currency")
    status = _resolve_column(dataframe, "status")
    refund = _resolve_column(dataframe, "is refund")
    chargeback = _resolve_column(dataframe, "is chargeback")

    if not q:
        return {"supported": False, "operation": "empty_question",
                "result": "Enter a question."}

    if any(x in q for x in ("total amount", "total transaction amount", "revenue", "sum of", "total value", "sales")) and amount:
        groups = [merchant] if merchant else []
        if currency and currency not in groups and (
            "currency" in q or dataframe[currency].nunique() > 1
        ):
            groups.append(currency)
        work = dataframe.copy()
        work[amount] = pd.to_numeric(work[amount], errors="coerce")
        if groups:
            result = work.groupby(groups, dropna=False)[amount].agg(
                transaction_count="count", total_amount="sum"
            ).reset_index().sort_values("total_amount", ascending=False)
            return _out("grouped_sum", {
                "group_columns": groups, "rows": _safe_records(result)
            })

    outcome = None
    mask = None
    if "chargeback" in q:
        outcome = "chargeback"
        if chargeback:
            mask = _flag_mask(dataframe[chargeback], outcome)
    elif "refund" in q or "refunded" in q:
        outcome = "refund"
        if refund:
            mask = _flag_mask(dataframe[refund], outcome)
    elif any(x in q for x in ("failed", "failure")) and status:
        outcome = "failed"
        mask = _status_mask(dataframe, status, "Failed")

    group = None
    if merchant and any(x in q for x in ("merchant", "merchants")):
        group = merchant
    elif "payment method" in q:
        group = _resolve_column(dataframe, "payment method")
    elif "country" in q or "countries" in q:
        group = _resolve_column(dataframe, "country")
    elif "category" in q:
        group = _resolve_column(dataframe, "category")

    if mask is not None:
        count = int(mask.sum())
        rate = round(count * 100 / len(dataframe), 2)
        if group is not None:
            work = dataframe.copy()
            work["_flag"] = mask.astype(int).values
            result = work.groupby(group, dropna=False).agg(
                transaction_count=(group, "size"),
                outcome_count=("_flag", "sum")
            ).reset_index()
            result["rate_percent"] = (
                result["outcome_count"] / result["transaction_count"] * 100
            ).round(2)
            result = result.sort_values("outcome_count", ascending=False)
            return _out("combined_outcome_group", {
                "outcome": outcome, "count": count,
                "denominator": len(dataframe), "rate_percent": rate,
                "group_column": group, "rows": _safe_records(result),
                "ranking_metric": "count"
            })
        return _out("overall_rate", {
            "outcome": outcome, "count": count,
            "denominator": len(dataframe), "rate_percent": rate
        })

    return _unsupported()
