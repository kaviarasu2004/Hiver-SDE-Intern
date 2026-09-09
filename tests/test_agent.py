"""
Offline smoke tests — no API key, no network. Run: pytest tests/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from ingest import build_pairs


def test_build_pairs_matches_int_and_float_ids():
    # regression test for the int/float id-matching bug found during dev:
    # tweet_id read as int64 ("1"), in_response_to_tweet_id read as float64
    # ("1.0") -> naive str merge silently drops every row.
    df = pd.DataFrame(
        {
            "tweet_id": [1, 2],
            "author_id": ["cust001", "SpotifyCares"],
            "text": ["help my app crashed", "try reinstalling, that usually fixes it"],
            "in_response_to_tweet_id": [float("nan"), 1.0],
            "created_at": ["Mon Jan 01 10:00:00 +0000 2024", "Mon Jan 01 10:05:00 +0000 2024"],
        }
    )
    pairs = build_pairs(df, "SpotifyCares")
    assert len(pairs) == 1
    assert pairs.iloc[0]["customer_text"] == "help my app crashed"
    assert "reinstalling" in pairs.iloc[0]["brand_reply"]


def test_build_pairs_drops_brand_to_brand_rows():
    df = pd.DataFrame(
        {
            "tweet_id": [1, 2],
            "author_id": ["SpotifyCares", "SpotifyCares"],
            "text": ["internal note", "reply to internal note"],
            "in_response_to_tweet_id": [float("nan"), 1.0],
            "created_at": ["Mon Jan 01 10:00:00 +0000 2024", "Mon Jan 01 10:05:00 +0000 2024"],
        }
    )
    pairs = build_pairs(df, "SpotifyCares")
    assert len(pairs) == 0
