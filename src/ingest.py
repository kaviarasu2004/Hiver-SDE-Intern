"""
ingest.py — reconstruct (customer_tweet, brand_reply) pairs from the
Kaggle "Customer Support on Twitter" dataset (twcs.csv).

Usage:
    python src/ingest.py --input data/twcs.csv --brand SpotifyCares \
        --out data/spotify_pairs.csv --limit 20000
"""
import argparse
import pandas as pd


def build_pairs(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    df = df.copy()
    # normalize both id columns through float -> nullable Int64 -> str so
    # "1" (int col) and "1.0" (float col, from NaNs) match consistently
    df["tweet_id"] = pd.array(df["tweet_id"], dtype="Int64").astype(str)
    df["in_response_to_tweet_id"] = (
        pd.to_numeric(df["in_response_to_tweet_id"], errors="coerce")
        .astype("Int64")
        .astype(str)
    )

    brand_replies = df[df["author_id"] == brand].copy()

    pairs = brand_replies.merge(
        df,
        left_on="in_response_to_tweet_id",
        right_on="tweet_id",
        suffixes=("_reply", "_inbound"),
    )

    # keep only genuine customer->brand turns (inbound not itself the brand)
    pairs = pairs[pairs["author_id_inbound"] != brand]

    keep = [
        "tweet_id_inbound",
        "text_inbound",
        "text_reply",
        "created_at_inbound",
        "author_id_inbound",
    ]
    out = pairs[keep].rename(
        columns={
            "tweet_id_inbound": "tweet_id",
            "text_inbound": "customer_text",
            "text_reply": "brand_reply",
            "created_at_inbound": "created_at",
            "author_id_inbound": "customer_id",
        }
    )

    # basic cleaning: drop empty/duplicate, strip @mentions boilerplate noise
    out = out.dropna(subset=["customer_text", "brand_reply"])
    out = out.drop_duplicates(subset=["tweet_id"])
    out = out[out["customer_text"].str.len() > 3]
    return out.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to twcs.csv")
    ap.add_argument("--brand", default="SpotifyCares", help="author_id of brand")
    ap.add_argument("--out", required=True, help="output pairs csv")
    ap.add_argument("--limit", type=int, default=None, help="cap rows read (fast local test)")
    args = ap.parse_args()

    df = pd.read_csv(args.input, nrows=args.limit)
    pairs = build_pairs(df, args.brand)
    pairs.to_csv(args.out, index=False)
    print(f"[ingest] {len(pairs)} pairs -> {args.out}")


if __name__ == "__main__":
    main()
