import html
import os
import re
import sys
import unicodedata

import pandas as pd

INPUT_FILE = "ronaldo-top.csv"
OUT_DIR = "clean"
MAIN_OUT = f"{OUT_DIR}/ronaldo_clean.csv"
TEXT_ONLY_OUT = f"{OUT_DIR}/ronaldo_text_clean.csv"
REVIEW_OUT = f"{OUT_DIR}/ronaldo_manual_review.csv"
REPORT_OUT = f"{OUT_DIR}/preprocessing_report.txt"

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

URL_RE = re.compile(r"(?:https?://|www\.)\S+|\bt\.co/\S+", re.I)
MENTION_RE = re.compile(r"@\w+")
INVISIBLE_RE = re.compile("[​‌‍⁠﻿ ]")
CURLY = str.maketrans(
    {"‘": "'", "’": "'", "“": '"', "”": '"'}
)
TOKEN_RE = re.compile(
    r"#[a-z0-9_]+"
    r"|(?:[a-z0-9]|(?<=\d)\.(?=\d))+"
)


def clean_text(s: str) -> str:
    """Clean a tweet into text_clean (see preprocessing (1).md, section 5)."""
    # 1. Normalize Unicode (NFKC) + curly quotes -> straight quotes,
    #    strip accents (NFKC then drop combining marks).
    s = unicodedata.normalize("NFKC", s).translate(CURLY)
    s = "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )
    # Remove invisible/zero-width characters.
    s = INVISIBLE_RE.sub(" ", s)
    # 2. Decode encoded strings (& > < " ...), possibly nested.
    s = re.sub(r"&(amp|lt|gt|quot|apos|nbsp)(?!;)", r"&\1;", s)
    prev = None
    while prev != s:
        prev, s = s, html.unescape(s)
    # 3. Remove URLs (before punctuation removal).
    s = URL_RE.sub(" ", s)
    # 4. Remove @mentions, then lowercase.
    s = MENTION_RE.sub(" ", s).lower()
    # 6. Commas between digits are removed with no space ("1,000" -> "1000").
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    # 5/7. Tokenize: keeps hashtags, letters/digits and digit.digit periods;
    #      apostrophes, %, other symbols, emojis and non-ASCII are dropped and
    #      act as a separator.
    # 8. Whitespace cleanup happens implicitly via " ".join().
    return " ".join(TOKEN_RE.findall(s))


# ---------------------------------------------------------------------------
# Likes validation
# ---------------------------------------------------------------------------

LIKE_SUFFIX = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}


def parse_likes(value) -> "int | None":
    """Parse favorite_count into a non-negative int, else None."""
    if pd.isna(value):
        return None  # missing stays null
    s = str(value).strip().replace(",", "").replace(" ", "")
    if not s:
        return None
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([kmb]?)", s, re.I)
    if not m:
        return None  # unparseable -> null (counted as invalid)
    n = float(m.group(1)) * LIKE_SUFFIX.get(m.group(2).lower(), 1)
    if n < 0:
        return None  # negative -> null
    return int(round(n))


# ---------------------------------------------------------------------------
# Relevance classification
# ---------------------------------------------------------------------------

RONALDO_KEYWORDS = ("cristiano", "ronaldo", "cr7", "cr 7")
FOOTBALL_CONTEXT = (
    "messi",
    "portugal",
    "al nassr",
    "real madrid",
    "manchester united",
    "juventus",
    "world cup",
    "football",
    "goat",
    "argentina",
    "national team",
    "retire",
    "fifa",
    "uefa",
)


def classify_relevance(text_clean: str) -> str:
    """Label a row relevant / review / irrelevant based on text_clean."""
    t = f" {text_clean} "  # pad so 'cr 7' and word checks behave
    if any(k in t for k in RONALDO_KEYWORDS):
        return "relevant"
    if "nazario" in t or any(k in t for k in FOOTBALL_CONTEXT):
        return "review"
    if len(text_clean.split()) <= 8:
        return "review"
    return "irrelevant"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    report: list[str] = []

    def say(line: str = "") -> None:
        print(line)
        report.append(line)

    # ---- Load -------------------------------------------------------------
    df = pd.read_csv(INPUT_FILE, dtype={"id_str": str}, encoding="utf-8")
    # Keep other *_str id-ish columns as strings too so CSV round-trips safely.
    initial_count = len(df)
    say(f"Initial row count: {initial_count}")

    # Never touch the original file: work on a copy.
    df = df.copy()
    df["full_text"] = df["full_text"].fillna("")

    # ---- Step 5 first: build text_clean (needed for dedup & relevance) -----
    df["text_clean"] = df["full_text"].map(
        lambda s: clean_text(s) if isinstance(s, str) else ""
    )

    # ---- 1. Remove duplicates ----------------------------------------------
    df["created_at"] = df["created_at"].astype(str)
    df = df.sort_values("created_at", kind="stable")
    dup_by_id = df.duplicated(subset="id_str").sum()
    df = df.drop_duplicates(subset="id_str", keep="first")
    say(f"Duplicates removed by id_str: {dup_by_id}")
    dup_by_text = df.duplicated(subset="text_clean").sum()
    df = df.drop_duplicates(subset="text_clean", keep="first")
    say(f"Duplicates removed by text_clean: {dup_by_text}")
    after_dedup = len(df)
    say(f"Rows after dedup: {after_dedup}")

    # ---- 2. Missing values --------------------------------------------------
    say("")
    say("Missing values per column (after dedup):")
    missing = df.replace("", pd.NA).isna().sum()
    for col, n in missing.items():
        say(f"  {col}: {n}")

    empty_fulltext = (df["full_text"].astype(str).str.strip() == "").sum()
    df = df[df["full_text"].astype(str).str.strip() != ""]
    empty_text_clean = (df["text_clean"].astype(str).str.strip() == "").sum()
    df = df[df["text_clean"].astype(str).str.strip() != ""]
    say(f"Rows with empty full_text removed: {empty_fulltext}")
    say(f"Rows with empty text_clean removed: {empty_text_clean}")
    after_missing = len(df)
    say(f"Rows after missing-value removal: {after_missing}")

    # ---- 4. Validate likes (before relevance, so all rows get valid counts) -
    likes_raw = df["favorite_count"]
    df["favorite_count"] = likes_raw.map(parse_likes)
    invalid_likes = int(df["favorite_count"].isna().sum())
    say("")
    say(f"Invalid/missing favorite_count (null) rows: {invalid_likes}")

    # ---- 3. Topic relevance -------------------------------------------------
    df["relevance"] = df["text_clean"].map(classify_relevance)
    counts = df["relevance"].value_counts()
    n_relevant = int(counts.get("relevant", 0))
    n_review = int(counts.get("review", 0))
    n_irrelevant = int(counts.get("irrelevant", 0))
    say("")
    say("Relevance results:")
    say(f"  relevant:   {n_relevant}")
    say(f"  review:     {n_review}")
    say(f"  irrelevant: {n_irrelevant}")

    main_df = df[df["relevance"] == "relevant"].copy()
    review_df = df[df["relevance"] == "review"].copy()
    say(f"Final row count of main file (relevant): {len(main_df)}")

    # ---- Output files --------------------------------------------------------
    main_df.to_csv(MAIN_OUT, index=False, encoding="utf-8-sig")
    review_df.to_csv(REVIEW_OUT, index=False, encoding="utf-8-sig")
    # text_clean-only file (relevant rows).
    main_df[["text_clean"]].to_csv(TEXT_ONLY_OUT, index=False, encoding="utf-8-sig")

    # ---- Validation ----------------------------------------------------------
    say("")
    say("Validation:")

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))
        say(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))

    check(
        "No duplicate id_str in main file",
        not main_df["id_str"].duplicated().any(),
        f"{int(main_df['id_str'].duplicated().sum())} duplicates",
    )
    ok_empty = (
        (main_df["full_text"].astype(str).str.strip() != "").all()
        and (main_df["text_clean"].astype(str).str.strip() != "").all()
    )
    check("No empty full_text or text_clean in main file", ok_empty)

    likes_series = main_df["favorite_count"]
    likes_ok = likes_series.isna().all() or (
        likes_series.dropna().map(lambda v: isinstance(v, (int, float)) and v >= 0).all()
    )
    check(
        "favorite_count null or non-negative integer",
        bool(likes_ok),
        f"{int(likes_series.isna().sum())} nulls",
    )

    # full_text unchanged & text_clean separate column
    orig = pd.read_csv(INPUT_FILE, dtype={"id_str": str}, encoding="utf-8")
    orig_map = orig.set_index("id_str")["full_text"]
    common_ids = main_df["id_str"].isin(orig_map.index).all()
    unchanged = True
    for _, row in main_df.iterrows():
        if orig_map.get(row["id_str"]) != row["full_text"]:
            unchanged = False
            break
    check("full_text unchanged & text_clean is separate column", bool(common_ids and unchanged))

    # text_clean content checks
    bad_patterns = {
        "'&' or encoded strings": re.compile(r"&|&\w+;", re.I),
        "uppercase letters": re.compile(r"[A-Z]"),
        "commas": re.compile(r","),
        "periods except between digits": re.compile(r"(?<!\d)\.|\.(?!\d)"),
        "apostrophes or %": re.compile(r"['%]"),
        "step-7 symbols": re.compile(r'[*;:!?"()/&+=$]'),
        "URLs": re.compile(r"(?:https?://|www\.)|\bt\.co/", re.I),
    }
    for label, pattern in bad_patterns.items():
        bad_rows = main_df["text_clean"].astype(str).str.contains(pattern, regex=True).sum()
        check(f"text_clean free of {label}", bad_rows == 0, f"{int(bad_rows)} rows")

    # Hashtags preserved (lowercased)
    HASHTAG_RE = re.compile(r"#\w+")
    lost_hashtags = 0
    for _, row in main_df.iterrows():
        orig_tags = {t.lower() for t in HASHTAG_RE.findall(str(row["full_text"]))}
        clean_tags = set(HASHTAG_RE.findall(str(row["text_clean"])))
        lost_hashtags += len(orig_tags - clean_tags)
    check(
        "Every hashtag from full_text kept (lowercased)",
        lost_hashtags == 0,
        f"{lost_hashtags} lost",
    )

    # Row-count reconciliation: removed = dups + empties + irrelevant
    removed_total = (initial_count - after_dedup) + n_irrelevant
    check(
        "Row count: initial = main + review + removed",
        initial_count == len(main_df) + len(review_df) + removed_total,
        f"initial={initial_count}, main={len(main_df)}, review={len(review_df)}, "
        f"removed(dups+irrelevant)={removed_total}",
    )

    # ---- Report --------------------------------------------------------------
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    n_fail = sum(1 for _, ok, _ in checks if not ok)
    say("")
    if n_fail:
        say(f"{n_fail} validation check(s) FAILED - see details above.")
        sys.exit(1)
    say("All validation checks passed.")
    say(f"Wrote {MAIN_OUT}, {TEXT_ONLY_OUT}, {REVIEW_OUT}, {REPORT_OUT}")


if __name__ == "__main__":
    main()
