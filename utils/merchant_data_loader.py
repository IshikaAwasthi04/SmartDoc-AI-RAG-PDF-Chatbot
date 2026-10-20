from io import BytesIO, StringIO
from pathlib import Path
import csv

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def _validate_column_names(columns):
    """Reject blank or duplicate column names after trimming whitespace."""
    normalized = [str(column).strip() for column in columns]

    if any(not column for column in normalized):
        raise ValueError(
            "One or more columns have empty names. "
            "Rename those columns and upload the file again."
        )

    seen = set()
    duplicates = []
    for column in normalized:
        key = column.casefold()
        if key in seen and column not in duplicates:
            duplicates.append(column)
        seen.add(key)

    if duplicates:
        raise ValueError(
            "The file contains duplicate column names. "
            "Rename them so every column is unique."
        )

    return normalized


def _load_csv(file_bytes):
    """Load CSV while checking the original header before pandas renames it."""
    try:
        csv_text = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        # Preserve support for common legacy encodings.
        csv_text = file_bytes.decode("latin-1")

    try:
        header = next(csv.reader(StringIO(csv_text)), [])
    except csv.Error as exc:
        raise ValueError(f"Could not read the CSV header: {exc}") from exc

    if not header:
        raise ValueError("The uploaded file contains no readable data.")

    # Check the original header before pandas can rename duplicate fields.
    normalized_header = _validate_column_names(header)

    try:
        dataframe = pd.read_csv(StringIO(csv_text))
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The uploaded file contains no readable data.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Could not read the uploaded file: {exc}") from exc

    # Keep the normalized names consistent with the validated source header.
    if len(dataframe.columns) != len(normalized_header):
        raise ValueError("The CSV header does not match the number of data columns.")

    dataframe.columns = normalized_header
    return dataframe


def load_merchant_data(uploaded_file):
    """
    Load a CSV or Excel file uploaded through Streamlit.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        pandas.DataFrame: Loaded tabular data.

    Raises:
        ValueError: If the file type or contents are invalid.
    """
    if uploaded_file is None:
        raise ValueError("Please upload a CSV or Excel file.")

    filename = Path(uploaded_file.name)
    extension = filename.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported file type. Upload a CSV, XLSX, or XLS file."
        )

    file_bytes = uploaded_file.getvalue()
    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    try:
        if extension == ".csv":
            dataframe = _load_csv(file_bytes)
        elif extension == ".xlsx":
            dataframe = pd.read_excel(BytesIO(file_bytes), engine="openpyxl")
        else:
            dataframe = pd.read_excel(BytesIO(file_bytes), engine="xlrd")

    except ValueError:
        raise
    except ImportError as exc:
        raise ValueError(
            f"Missing dependency required to read this file: {exc}"
        ) from exc
    except Exception as exc:
        raise ValueError(f"File loading failed: {exc}") from exc

    if dataframe.empty:
        raise ValueError("The uploaded file has no data rows.")

    # Normalize and validate Excel headers too.
    dataframe.columns = _validate_column_names(dataframe.columns)

    # Remove rows that contain no data at all.
    dataframe = dataframe.dropna(how="all").reset_index(drop=True)

    if dataframe.empty:
        raise ValueError("The file contains no usable data rows.")

    return dataframe
