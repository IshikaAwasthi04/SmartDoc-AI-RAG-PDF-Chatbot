
import pandas as pd


def get_data_overview(dataframe):
    """
    Return a basic overview of the uploaded tabular data.
    """
    if dataframe is None or dataframe.empty:
        raise ValueError("No data is available to analyze.")

    return {
        "row_count": int(dataframe.shape[0]),
        "column_count": int(dataframe.shape[1]),
        "columns": dataframe.columns.tolist(),
        "missing_values": {
            str(column): int(count)
            for column, count in dataframe.isna().sum().items()
        },
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "data_types": {
            str(column): str(dtype)
            for column, dtype in dataframe.dtypes.items()
        },
    }


def get_numeric_summary(dataframe, column):
    """
    Calculate descriptive statistics for a numeric column.
    """
    _validate_column(dataframe, column)

    numeric_values = pd.to_numeric(
        dataframe[column],
        errors="coerce"
    ).dropna()

    if numeric_values.empty:
        raise ValueError(
            f"Column '{column}' has no usable numeric values."
        )

    return {
        "column": column,
        "count": int(numeric_values.count()),
        "sum": float(numeric_values.sum()),
        "mean": float(numeric_values.mean()),
        "median": float(numeric_values.median()),
        "minimum": float(numeric_values.min()),
        "maximum": float(numeric_values.max()),
    }


def filter_rows(dataframe, column, operator, value):
    """
    Filter rows using a supported comparison operator.

    Supported operators: ==, !=, >, >=, <, <=, contains
    """
    _validate_column(dataframe, column)

    series = dataframe[column]

    if operator == "contains":
        mask = series.astype("string").str.contains(
            str(value),
            case=False,
            na=False,
            regex=False
        )

    elif operator in {">", ">=", "<", "<="}:
        numeric_series = pd.to_numeric(
            series,
            errors="coerce"
        )

        try:
            comparison_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Numeric comparisons require a numeric value."
            ) from exc

        if operator == ">":
            mask = numeric_series > comparison_value
        elif operator == ">=":
            mask = numeric_series >= comparison_value
        elif operator == "<":
            mask = numeric_series < comparison_value
        else:
            mask = numeric_series <= comparison_value

    elif operator == "==":
        mask = series.astype("string") == str(value)

    elif operator == "!=":
        mask = series.astype("string") != str(value)

    else:
        raise ValueError(
            "Unsupported operator. Use ==, !=, >, >=, <, <=, or contains."
        )

    return dataframe.loc[mask.fillna(False)].copy()


def get_top_rows(dataframe, column, count=5, ascending=False):
    """
    Return the top or bottom rows according to a column.
    """
    _validate_column(dataframe, column)

    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("Count must be a positive integer.")

    if count > 100:
        raise ValueError("Count cannot exceed 100.")

    sort_values = dataframe[column]

    # Sort numerically when the column contains numeric data.
    numeric_values = pd.to_numeric(sort_values, errors="coerce")

    if numeric_values.notna().sum() > 0:
        sortable = dataframe.assign(
            _sort_value=numeric_values
        )

        result = sortable.sort_values(
            "_sort_value",
            ascending=ascending,
            na_position="last"
        ).head(count)

        return result.drop(columns=["_sort_value"])

    return dataframe.sort_values(
        column,
        ascending=ascending,
        na_position="last"
    ).head(count)


def get_value_counts(dataframe, column):
    """
    Count the frequency of each distinct value in a column.
    """
    _validate_column(dataframe, column)

    counts = dataframe[column].value_counts(dropna=False)

    return {
        str(value): int(count)
        for value, count in counts.items()
    }


def _validate_column(dataframe, column):
    """
    Ensure the requested column exists.
    """
    if dataframe is None or dataframe.empty:
        raise ValueError("No data is available to analyze.")

    if column not in dataframe.columns:
        raise ValueError(
            f"Column '{column}' was not found. "
            f"Available columns: {', '.join(map(str, dataframe.columns))}"
        )
