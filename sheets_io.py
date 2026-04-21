"""Google Sheets I/O for the Habermas Machine.

Public helper for fetching question + opinions (or critiques) from a
publicly-viewable Google Sheet via its CSV export endpoint. No OAuth
required — the sheet must be set to "Anyone with the link can view".
"""

import io
import re


def fetch_from_google_sheets(
    sheet_url: str,
    opinion_column: str = 'B',
    question_column: str = 'A',
) -> tuple[str | None, list[str]]:
  """Fetches a question + opinions list from a public Google Sheet.

  Args:
    sheet_url: Full URL of the Google Sheet (the normal browser URL with
      `/spreadsheets/d/<id>/...`).
    opinion_column: Column letter containing opinions (or critiques). Defaults
      to 'B'.
    question_column: Column letter containing the question; first row only.
      Pass an empty string to skip. Defaults to 'A'.

  Returns:
    (question, opinions). `question` is None if not present or blank.

  Raises:
    ImportError: if pandas/requests are not installed.
    ValueError: if the URL can't be parsed or the column doesn't exist.
    Exception: wraps any network / parsing failure with context.
  """
  try:
    import pandas as pd
    import requests
  except ImportError as e:
    raise ImportError(
        'Google Sheets integration requires: pip install pandas requests'
    ) from e

  match = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', sheet_url)
  if not match:
    raise ValueError('Invalid Google Sheets URL')
  sheet_id = match.group(1)

  csv_url = (
      f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv'
  )

  try:
    response = requests.get(csv_url, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
  except Exception as e:
    raise Exception(f'Error fetching from Google Sheets: {e}') from e

  opinion_idx = ord(opinion_column.upper()) - ord('A')
  if opinion_idx >= len(df.columns):
    raise ValueError(f'Column {opinion_column} not found in sheet')

  opinions: list[str] = []
  for i in range(len(df)):
    value = str(df.iloc[i, opinion_idx]).strip()
    if value and value != 'nan':
      opinions.append(value)

  question: str | None = None
  if question_column and len(df.columns) > 0 and len(df) > 0:
    q_idx = ord(question_column.upper()) - ord('A')
    if q_idx < len(df.columns):
      candidate = str(df.iloc[0, q_idx]).strip()
      if candidate and candidate != 'nan':
        question = candidate

  return question, opinions
