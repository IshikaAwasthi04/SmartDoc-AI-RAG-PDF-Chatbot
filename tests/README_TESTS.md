# Merchant assistant tests

Copy the `tests/` directory into the root of your project.

In PowerShell, from the repository root with the project virtual environment active:

```powershell
python -m pip install pytest pandas openpyxl
python -m pytest tests -q
```

These tests assume the module names and public function signatures currently used by the project:
- `utils.merchant_data_loader.load_merchant_data`
- `utils.merchant_analytics.get_data_overview`
- `utils.merchant_analytics.get_numeric_summary`
- `utils.merchant_analytics.filter_rows`
- `utils.merchant_analytics.get_top_rows`
- `utils.merchant_analytics.get_value_counts`
- `utils.merchant_chatbot.analyze_merchant_question`
- `utils.merchant_chatbot.answer_merchant_question`

If a test fails because an analytics function returns a different documented shape, inspect that failure rather than weakening the assertion.
