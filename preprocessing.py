import argparse
import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path


def remove_symbols(text):
    cleaned = []
    for ch in text:
        if ch in {"\u200d", "\ufe0f"}:
            continue
        category = unicodedata.category(ch)
        if category.startswith("S"):
            continue
        cleaned.append(ch)
    return "".join(cleaned)


def normalize_whitespace(text):
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(value):
    text = normalize_whitespace(value)
    if not text:
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = remove_symbols(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_like(value):
    if value is None:
        return None

    raw = str(value).strip()
    if raw == "":
        return None

    raw = raw.replace(",", "")
    if re.fullmatch(r"\d+", raw):
        parsed = int(raw)
        return parsed if parsed >= 0 else None

    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kKmMbB])", raw)
    if match:
        number = float(match.group(1))
        multiplier = {"k": 1000, "K": 1000, "m": 1_000_000, "M": 1_000_000,
                     "b": 1_000_000_000, "B": 1_000_000_000}
        parsed = int(round(number * multiplier[match.group(2)]))
        return parsed if parsed >= 0 else None

    return None


def normalize_for_duplicate_check(value):
    text = clean_text(value)
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def classify_topic(text):
    text = normalize_whitespace(text).lower()
    if not text:
        return "irrelevant"

    if re.search(r"\b(?:cristiano\s*ronaldo|ronaldo|cr7|cristiano)\b", text):
        return "relevant"

    football_terms = [
        "football", "soccer", "goal", "goals", "match", "team", "portugal",
        "portuguese", "messi", "fifa", "transfer", "club", "league", "national team"
    ]
    if any(term in text for term in football_terms):
        return "manual_review"

    return "irrelevant"


def get_missing_value_counts(rows, columns):
    counts = {}
    for column in columns:
        counts[column] = sum(1 for row in rows if row.get(column) is None or str(row.get(column)).strip() == "")
    return counts


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def validate_rows(rows):
    seen_ids = set()
    for row in rows:
        tweet_id = str(row.get("id_str", "")).strip()
        if tweet_id:
            if tweet_id in seen_ids:
                raise ValueError(f"Duplicate tweet ID found: {tweet_id}")
            seen_ids.add(tweet_id)

        cleaned = str(row.get("text_clean", "")).strip()
        if cleaned == "":
            raise ValueError(f"Empty cleaned text found for tweet ID: {tweet_id or 'unknown'}")

        like_value = row.get("favorite_count")
        if like_value is not None and str(like_value).strip() != "":
            if parse_like(like_value) is None:
                raise ValueError(f"Invalid non-missing like count: {like_value!r}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess the Ronaldo tweet dataset.")
    parser.add_argument("--input", default="ronaldo-top.csv", help="Input CSV path.")
    parser.add_argument("--output", default="ronaldo-top/ronaldo-top_preprocessed.csv", help="Main processed output CSV path.")
    parser.add_argument("--manual-review", default="ronaldo-top/ronaldo-top_manual_review.csv", help="Ambiguous/manual review output CSV path.")
    parser.add_argument("--report", default="ronaldo-top/preprocessing-top_report.txt", help="Text report path.")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    input_path = (base_dir / args.input).resolve()
    output_path = (base_dir / args.output).resolve()
    manual_review_path = (base_dir / args.manual_review).resolve()
    report_path = (base_dir / args.report).resolve()

    with open(input_path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    initial_count = len(rows)
    missing_before = get_missing_value_counts(rows, fieldnames)
    removed_duplicates = 0
    empty_text_removed = 0
    invalid_likes = 0
    relevance_counter = Counter({"relevant": 0, "irrelevant": 0, "manual_review": 0})

    seen_ids = set()
    seen_texts = set()
    cleaned_rows = []

    for row in rows:
        tweet_id = (row.get("id_str") or "").strip()
        tweet_text = row.get("full_text") or ""
        duplicate_key = normalize_for_duplicate_check(tweet_text)

        if tweet_id and tweet_id in seen_ids:
            removed_duplicates += 1
            continue
        if duplicate_key and duplicate_key in seen_texts:
            removed_duplicates += 1
            continue

        if tweet_id:
            seen_ids.add(tweet_id)
        if duplicate_key:
            seen_texts.add(duplicate_key)

        text_original = tweet_text
        if not text_original or not text_original.strip():
            empty_text_removed += 1
            continue

        original_like = row.get("favorite_count")
        parsed_like = parse_like(original_like)
        if original_like is not None and str(original_like).strip() != "" and parsed_like is None:
            invalid_likes += 1
            row["favorite_count"] = ""
        elif original_like is not None and str(original_like).strip() != "":
            row["favorite_count"] = parsed_like
        else:
            row["favorite_count"] = ""

        row["text_original"] = text_original
        row["text_clean"] = clean_text(text_original)
        row["full_text"] = text_original

        classification = classify_topic(text_original)
        relevance_counter[classification] += 1
        cleaned_rows.append((classification, row))

    main_rows = []
    manual_rows = []
    for classification, row in cleaned_rows:
        if classification == "relevant":
            main_rows.append(row)
        elif classification == "manual_review":
            manual_rows.append(row)
        else:
            continue

    final_fields = [
        "conversation_id_str", "created_at", "favorite_count", "id_str", "image_url",
        "in_reply_to_screen_name", "lang", "location", "quote_count", "reply_count",
        "retweet_count", "tweet_url", "user_id_str", "username", "text_original", "text_clean"
    ]

    for row in main_rows:
        row.pop("full_text", None)
    for row in manual_rows:
        row.pop("full_text", None)

    write_csv(output_path, final_fields, main_rows)
    write_csv(manual_review_path, final_fields, manual_rows)

    validate_rows(main_rows)

    final_row_count = len(main_rows)
    report_lines = [
        "Preprocessing Report",
        "====================",
        f"Initial row count: {initial_count}",
        f"Removed duplicates: {removed_duplicates}",
        f"Removed empty text rows: {empty_text_removed}",
        "",
        "Missing values by column:",
    ]

    for column, count in missing_before.items():
        if count > 0:
            report_lines.append(f"- {column}: {count}")

    report_lines.extend([
        "",
        "Relevance results:",
        f"- relevant: {relevance_counter['relevant']}",
        f"- manual_review: {relevance_counter['manual_review']}",
        f"- irrelevant: {relevance_counter['irrelevant']}",
        f"- invalid likes: {invalid_likes}",
        f"Final row count: {final_row_count}",
    ])

    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(report_lines) + "\n")

    print(f"Processed {initial_count} rows into {final_row_count} relevant rows.")
    print(f"Saved main dataset to: {output_path}")
    print(f"Saved manual review rows to: {manual_review_path}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
