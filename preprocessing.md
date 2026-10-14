# Task: Preprocess Ronaldo Tweet in X (Twitter) Dataset

Act as a data preprocessing agent. Process the input CSV (`ronaldo.csv`) according to the requirements below. Execute the preprocessing, validate the output files, and report the results.

## General Rules

- Never modify the original dataset. Never fabricate missing data.
- Keep all original column names as they are (`id_str`, `full_text`, `favorite_count`, and the rest). Do not rename or drop any column.
- `full_text` stays as the untouched original tweet. The cleaned version goes in a new column `text_clean`.
- Read the CSV with a proper CSV parser (pandas). Tweets contain line breaks, so the file has more lines than records. Read `id_str` as a string.
- Do not add any cleaning that is not listed here (no stopword removal, stemming, or spell correction).

## 1. Remove Duplicates

- Remove duplicates by `id_str`, then by `text_clean`. Keep one record each time (earliest `created_at`).
- Compare on `text_clean`, not on raw `full_text`. Raw text never matches because each tweet has a different t.co link.

## 2. Handle Missing Values

- Report missing values per column.
- Remove rows with empty `full_text`, and rows where `text_clean` ends up empty.
- Do not replace missing `favorite_count` with zero. A value of 0 is valid, not missing.

## 3. Check Topic Relevance

Topic: tweets about Cristiano Ronaldo. Judge on `text_clean`. Add a column `relevance` with one label per row.

- **relevant**: contains `cristiano`, `ronaldo`, `cr7`, or `cr 7`. Stays in the main dataset.
- **review**: no Ronaldo keyword but has football context (`messi`, `portugal`, `al nassr`, `real madrid`, `manchester united`, `juventus`, `world cup`, `football`, `goat`, `argentina`, `national team`, `retire`, `fifa`, `uefa`), or has 8 words or fewer, or mentions `nazario` (possible Ronaldo Nazario). Save separately.
- **irrelevant**: everything else (for example a clinical AI post, a baseball comment, a song lyric). Remove from the main dataset.

## 4. Validate Likes

- Column `favorite_count`. Must be a non-negative integer.
- Handle formats such as `1.2K` (1200) and `1.5M` (1500000). Strip commas and spaces.
- Negative or unparseable values become null and are counted as invalid.
- Missing stays null.

## 5. Clean Text

Build `text_clean` from `full_text` in this order:

1. **Normalize Unicode** (NFKC). Some tweets write the name in fancy bold letters that will not match otherwise. Convert curly quotes to straight quotes. Convert accented letters to plain letters.
2. **Remove encoded strings** such as `&amp;`, `&gt;`, `&lt;`, `&quot;`. Decode first, then the leftover `&` is removed in step 7.
3. **Remove URLs** (`http...`, `t.co/...`). Do this before removing punctuation.
4. **Remove @mentions.**
5. **Lowercase** everything.
6. **Remove all commas and periods.** Exception: a period between two digits (`2.5`) stays. A comma between digits (`1,000`) is removed without a space (`1000`).
7. **Remove unnecessary symbols**: `* ; : - ! ? " ( ) / & _ + = $`, emojis, and any other non-ASCII character. **Do not remove hashtags**: `#` and the word after it stay (`#CR7` becomes `#cr7`). Keep apostrophes inside words (`don't`) and `%` after a number (`26%`).
8. **Clean whitespace.** Line breaks and repeated spaces become one space. Trim both ends.

Replace removed symbols with a space, not with nothing, so words do not merge (`back-to-back` becomes `back to back`).

Reference implementation:

```python
import re, html, unicodedata

URL_RE = re.compile(r'(?:https?://|www\.)\S+|\bt\.co/\S+', re.I)
MENTION_RE = re.compile(r'@\w+')
INVISIBLE_RE = re.compile('[\u200b\u200c\u200d\u2060\ufeff\u00a0]')
CURLY = str.maketrans({'\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"'})
TOKEN_RE = re.compile(
    r"#[a-z0-9_]+"
    r"|(?:[a-z0-9]|(?<=\d)\.(?=\d)|(?<=[a-z])'(?=[a-z]))+(?:(?<=\d)%)?"
)

def clean_text(s):
    s = unicodedata.normalize('NFKC', s).translate(CURLY)
    s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    s = INVISIBLE_RE.sub(' ', s)
    s = re.sub(r'&(amp|lt|gt|quot|apos|nbsp)(?!;)', r'&\1;', s)
    prev = None
    while prev != s:
        prev, s = s, html.unescape(s)
    s = URL_RE.sub(' ', s)
    s = MENTION_RE.sub(' ', s).lower()
    s = re.sub(r'(?<=\d),(?=\d)', '', s)
    return ' '.join(TOKEN_RE.findall(s))
```

Example: `Ronaldo &amp; Messi: back-to-back, 1,000 fans! #CR7 https://t.co/x @user` becomes `ronaldo messi back to back 1000 fans #cr7`.

## Output Files

- `ronaldo_clean.csv`: relevant rows only. All original columns, plus `text_clean` and `relevance`.
- `ronaldo_manual_review.csv`: review rows, same columns.
- `preprocessing_report.txt`: see below.

Write CSVs with `utf-8-sig` encoding.

## Validation

Check the output files and confirm:

- No duplicate `id_str` in the main file.
- No empty `full_text` or `text_clean`.
- `favorite_count` is null or a non-negative integer in every row.
- `full_text` is unchanged from the original and `text_clean` is a separate column.
- `text_clean` has: no `&` or encoded strings, no uppercase, no commas, no periods except between digits, none of the symbols from step 7, no URLs.
- Every hashtag in the original tweet is still in `text_clean` (lowercased).
- Row count: initial = main + review + removed.

If any check fails, fix it and rerun.

## Report: `preprocessing_report.txt`

Summarize, with numbers computed from the data:

- Initial row count
- Duplicates removed (by `id_str`, by `text_clean`)
- Missing values per column and empty tweets removed
- Relevance results (relevant, review, irrelevant)
- Invalid likes
- Final row count of the main file
- Validation results (pass or fail per check)
