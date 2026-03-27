"""
Test suite for app_7.py — Telemarketing Dashboard
Author: Jefferson Luis Risso dos Santos

Tests cover the core data processing functions:
- multiselect_filter: categorical filtering logic
- convert_df: CSV serialization
- to_excel: Excel serialization
- load_data: file ingestion (CSV and Excel)

Each test validates not just that the code runs, but that it produces
the correct output — including edge cases and potential failure modes.
"""

import pytest
import pandas as pd
import numpy as np
from io import BytesIO, StringIO

# ── Import functions under test ─────────────────────────────────────────────
# We import only the pure functions (no Streamlit decorators in test context).
# The @st.cache_data decorators are transparent to pytest since we call the
# underlying functions directly via their __wrapped__ attribute when needed.
import importlib, sys, types

# Stub out streamlit so the module loads without a running Streamlit server
streamlit_stub = types.ModuleType("streamlit")
streamlit_stub.cache_data = lambda *a, **kw: (lambda f: f) if not a else a[0]
streamlit_stub.set_page_config = lambda **kw: None
sys.modules.setdefault("streamlit", streamlit_stub)

# Also stub heavy UI deps not needed for unit tests
for mod in ["PIL", "PIL.Image", "seaborn", "matplotlib", "matplotlib.pyplot"]:
    sys.modules.setdefault(mod, types.ModuleType(mod))

from app_7 import multiselect_filter, convert_df, to_excel, load_data


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal dataframe mimicking the bank marketing dataset structure."""
    return pd.DataFrame({
        "age":         [25, 35, 45, 55, 65],
        "job":         ["admin", "technician", "admin", "retired", "technician"],
        "marital":     ["single", "married", "married", "divorced", "single"],
        "default":     ["no", "no", "yes", "no", "no"],
        "housing":     ["yes", "no", "yes", "no", "yes"],
        "loan":        ["no", "yes", "no", "no", "yes"],
        "contact":     ["cellular", "telephone", "cellular", "cellular", "telephone"],
        "month":       ["may", "jun", "may", "aug", "jun"],
        "day_of_week": ["mon", "tue", "wed", "thu", "fri"],
        "y":           ["no", "yes", "no", "yes", "no"],
    })


# ── multiselect_filter ───────────────────────────────────────────────────────

class TestMultiselectFilter:

    def test_filter_single_value_returns_matching_rows(self, sample_df):
        """Filtering by a specific job should return only rows with that job."""
        result = multiselect_filter(sample_df, "job", ["admin"])
        assert list(result["job"].unique()) == ["admin"]
        assert len(result) == 2

    def test_filter_all_keyword_returns_full_dataframe(self, sample_df):
        """'all' in selection must bypass filtering and return the original data."""
        result = multiselect_filter(sample_df, "job", ["all"])
        assert len(result) == len(sample_df)

    def test_filter_all_with_other_values_still_returns_full_df(self, sample_df):
        """
        If 'all' is present alongside specific values, the full dataset should
        be returned — 'all' takes precedence. This is a common false-positive
        trap: a naive implementation might filter to only ['admin', 'all'].
        """
        result = multiselect_filter(sample_df, "job", ["admin", "all"])
        assert len(result) == len(sample_df)

    def test_filter_multiple_values(self, sample_df):
        """Filtering by multiple categories should include rows matching any of them."""
        result = multiselect_filter(sample_df, "job", ["admin", "retired"])
        assert set(result["job"].unique()) == {"admin", "retired"}
        assert len(result) == 3

    def test_filter_nonexistent_value_returns_empty(self, sample_df):
        """
        Filtering by a value not present in the column should return an empty
        DataFrame — not raise an error, not return the full dataset.
        """
        result = multiselect_filter(sample_df, "job", ["astronaut"])
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)

    def test_filter_resets_index(self, sample_df):
        """
        After filtering, the index should be reset starting from 0.
        A broken reset_index would cause downstream iloc/positional access to fail.
        """
        result = multiselect_filter(sample_df, "job", ["technician"])
        assert list(result.index) == list(range(len(result)))

    def test_filter_does_not_mutate_original(self, sample_df):
        """The original DataFrame must not be modified by the filter operation."""
        original_len = len(sample_df)
        multiselect_filter(sample_df, "job", ["admin"])
        assert len(sample_df) == original_len

    def test_filter_empty_selection_returns_empty(self, sample_df):
        """
        An empty selection list (no 'all', no values) should return an empty
        DataFrame. This edge case can cause silent errors if mishandled.
        """
        result = multiselect_filter(sample_df, "job", [])
        assert len(result) == 0

    def test_filter_preserves_all_columns(self, sample_df):
        """Filtering should not drop any columns from the dataset."""
        result = multiselect_filter(sample_df, "job", ["admin"])
        assert list(result.columns) == list(sample_df.columns)


# ── convert_df ──────────────────────────────────────────────────────────────

class TestConvertDf:

    def test_returns_bytes(self, sample_df):
        """Output must be bytes — required for st.download_button compatibility."""
        result = convert_df(sample_df)
        assert isinstance(result, bytes)

    def test_csv_is_parseable(self, sample_df):
        """The bytes output must be valid CSV that can be re-read into a DataFrame."""
        result = convert_df(sample_df)
        recovered = pd.read_csv(BytesIO(result))
        assert list(recovered.columns) == list(sample_df.columns)

    def test_csv_row_count_matches(self, sample_df):
        """Number of data rows in the CSV must equal the original DataFrame length."""
        result = convert_df(sample_df)
        recovered = pd.read_csv(BytesIO(result))
        assert len(recovered) == len(sample_df)

    def test_csv_excludes_index(self, sample_df):
        """
        The CSV must not include the DataFrame index as a column (index=False).
        Including the index would add an unexpected 'Unnamed: 0' column on re-read.
        """
        result = convert_df(sample_df)
        recovered = pd.read_csv(BytesIO(result))
        assert "Unnamed: 0" not in recovered.columns

    def test_csv_encoding_is_utf8(self, sample_df):
        """Output must be UTF-8 encoded — required for correct handling of accents."""
        result = convert_df(sample_df)
        decoded = result.decode("utf-8")
        assert "age" in decoded  # column name survives round-trip


# ── to_excel ────────────────────────────────────────────────────────────────

class TestToExcel:

    def test_returns_bytes(self, sample_df):
        """Excel output must be bytes for download button compatibility."""
        result = to_excel(sample_df)
        assert isinstance(result, bytes)

    def test_excel_is_parseable(self, sample_df):
        """Bytes output must be a valid Excel file readable by pandas."""
        result = to_excel(sample_df)
        recovered = pd.read_excel(BytesIO(result))
        assert list(recovered.columns) == list(sample_df.columns)

    def test_excel_row_count_matches(self, sample_df):
        """Row count in Excel output must match the source DataFrame."""
        result = to_excel(sample_df)
        recovered = pd.read_excel(BytesIO(result))
        assert len(recovered) == len(sample_df)

    def test_excel_sheet_name_is_sheet1(self, sample_df):
        """The worksheet must be named 'Sheet1' as specified in the implementation."""
        result = to_excel(sample_df)
        xl = pd.ExcelFile(BytesIO(result))
        assert "Sheet1" in xl.sheet_names

    def test_excel_excludes_index(self, sample_df):
        """
        Excel export must not write the DataFrame index as a column.
        A spurious index column would break downstream data consumers.
        """
        result = to_excel(sample_df)
        recovered = pd.read_excel(BytesIO(result))
        assert "Unnamed: 0" not in recovered.columns

    def test_empty_dataframe_exports_without_error(self):
        """
        Exporting an empty DataFrame (e.g., after filtering removes all rows)
        must succeed and produce a valid empty Excel file — not raise an exception.
        """
        empty_df = pd.DataFrame(columns=["age", "job", "y"])
        result = to_excel(empty_df)
        recovered = pd.read_excel(BytesIO(result))
        assert len(recovered) == 0
        assert list(recovered.columns) == ["age", "job", "y"]


# ── load_data ────────────────────────────────────────────────────────────────

class TestLoadData:

    def test_load_csv_semicolon_separator(self, sample_df):
        """
        The app uses sep=';' for CSV — the bank-additional.csv dataset uses
        semicolons, not commas. A comma-based read would produce a single-column
        DataFrame with all data concatenated in one field.
        """
        csv_bytes = sample_df.to_csv(index=False, sep=";").encode("utf-8")
        file_like = BytesIO(csv_bytes)
        result = load_data(file_like)
        assert list(result.columns) == list(sample_df.columns)
        assert len(result) == len(sample_df)

    def test_load_excel_file(self, sample_df):
        """load_data must fall back to read_excel when CSV parsing fails."""
        excel_bytes = to_excel(sample_df)
        file_like = BytesIO(excel_bytes)
        # Simulate CSV parse failure by passing Excel bytes — triggers except branch
        result = load_data(file_like)
        assert list(result.columns) == list(sample_df.columns)
        assert len(result) == len(sample_df)

    def test_loaded_dataframe_has_expected_columns(self, sample_df):
        """
        Critical regression test: all expected columns must survive the load.
        Missing columns would cause KeyError in all downstream filter operations.
        """
        csv_bytes = sample_df.to_csv(index=False, sep=";").encode("utf-8")
        result = load_data(BytesIO(csv_bytes))
        expected_cols = ["age", "job", "marital", "default", "housing",
                         "loan", "contact", "month", "day_of_week", "y"]
        for col in expected_cols:
            assert col in result.columns, f"Missing column after load: {col}"
