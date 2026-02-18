"""
data_processing.py — Load review.json, filter, and build SQLite databases.

Usage:
    python src/data_processing.py            # builds full DB + sample DB
    python src/data_processing.py --sample   # builds only the 5 000-row sample DB
"""

import json
import random
import sqlite3
import sys
import time
from pathlib import Path
from textblob import TextBlob

# Allow running both as module and as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import (
    DATA_DIR,
    DB_PATH,
    MAX_YEAR,
    MIN_YEAR,
    REVIEW_JSON_PATH,
    SAMPLE_DB_PATH,
    get_year,
    parse_date,
)

SCHEMA_PATH = DATA_DIR / "data_schema.sql"
SAMPLE_SIZE = 5500  # slightly more than 5 000 required
BATCH_SIZE = 10_000


# ── Schema helpers ─────────────────────────────────────────────────────────────

def _create_schema(conn: sqlite3.Connection) -> None:
    """Execute the DDL from data_schema.sql."""
    schema_sql = SCHEMA_PATH.read_text()
    conn.executescript(schema_sql)


def _drop_indexes(conn: sqlite3.Connection) -> None:
    """Drop indexes before bulk insert for speed."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
    ).fetchall()
    for (name,) in rows:
        conn.execute(f"DROP INDEX IF EXISTS {name}")
    conn.commit()


def _create_indexes(conn: sqlite3.Connection) -> None:
    """Re-create indexes after bulk insert."""
    schema_sql = SCHEMA_PATH.read_text()
    for line in schema_sql.splitlines():
        if line.strip().upper().startswith("CREATE INDEX"):
            conn.execute(line)
    conn.commit()


# ── Parsing one JSON line ──────────────────────────────────────────────────────

def _parse_review(raw: dict) -> dict | None:
    """
    Parse a single raw JSON record into (hotel_row, author_row, review_row).
    Returns None if the record should be skipped.
    """
    year = get_year(raw.get("date", ""))
    if year is None or year < MIN_YEAR or year > MAX_YEAR:
        return None

    ratings = raw.get("ratings", {})
    author = raw.get("author", {})

    # Filter: Must have overall rating
    if ratings.get("overall") is None:
        return None

    # Filter: Must have substantial text content (> 20 chars)
    text = raw.get("text", "")
    if not text or len(text) < 20:
        return None

    dt = parse_date(raw.get("date", ""))
    date_iso = dt.strftime("%Y-%m-%d") if dt else None
    month = dt.month if dt else None

    # Sentiment Analysis
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    return {
        "hotel_id": raw.get("offering_id"),
        "author": {
            "author_id": author.get("id"),
            "username": author.get("username"),
            "location": author.get("location"),
            "num_cities": author.get("num_cities"),
            "num_helpful_votes": author.get("num_helpful_votes"),
            "num_reviews": author.get("num_reviews"),
            "num_type_reviews": author.get("num_type_reviews"),
        },
        "review": {
            "review_id": raw.get("id"),
            "hotel_id": raw.get("offering_id"),
            "author_id": author.get("id"),
            "title": raw.get("title"),
            "text": text,
            "date": raw.get("date"),
            "date_parsed": date_iso,
            "year": year,
            "month": month,
            "date_stayed": raw.get("date_stayed"),
            "rating_service": ratings.get("service"),
            "rating_cleanliness": ratings.get("cleanliness"),
            "rating_overall": ratings.get("overall"),
            "rating_value": ratings.get("value"),
            "rating_location": ratings.get("location"),
            "rating_sleep_quality": ratings.get("sleep_quality"),
            "rating_rooms": ratings.get("rooms"),
            "num_helpful_votes": raw.get("num_helpful_votes", 0),
            "via_mobile": 1 if raw.get("via_mobile") else 0,
            "sentiment_polarity": polarity,
            "sentiment_subjectivity": subjectivity,
        },
    }


# ── Bulk insert ────────────────────────────────────────────────────────────────

def _insert_batch(
    conn: sqlite3.Connection,
    hotels: dict,
    authors: dict,
    reviews: list[dict],
) -> None:
    """Insert a batch of parsed records into the database."""
    # Hotels (upsert)
    hotel_rows = [(hid,) for hid in hotels if hid is not None]
    conn.executemany(
        "INSERT OR IGNORE INTO hotels (hotel_id) VALUES (?)", hotel_rows
    )

    # Authors (upsert)
    author_rows = [
        (
            a["author_id"], a["username"], a["location"],
            a["num_cities"], a["num_helpful_votes"],
            a["num_reviews"], a["num_type_reviews"],
        )
        for a in authors.values()
        if a["author_id"] is not None
    ]
    conn.executemany(
        """INSERT OR IGNORE INTO authors
           (author_id, username, location, num_cities,
            num_helpful_votes, num_reviews, num_type_reviews)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        author_rows,
    )

    # Reviews
    review_rows = [
        (
            r["review_id"], r["hotel_id"], r["author_id"],
            r["title"], r["text"], r["date"], r["date_parsed"],
            r["year"], r["month"], r["date_stayed"],
            r["rating_service"], r["rating_cleanliness"],
            r["rating_overall"], r["rating_value"],
            r["rating_location"], r["rating_sleep_quality"],
            r["rating_rooms"], r["num_helpful_votes"], r["via_mobile"],
            r["sentiment_polarity"], r["sentiment_subjectivity"],
        )
        for r in reviews
    ]
    conn.executemany(
        """INSERT OR IGNORE INTO reviews
           (review_id, hotel_id, author_id, title, text, date, date_parsed,
            year, month, date_stayed, rating_service, rating_cleanliness,
            rating_overall, rating_value, rating_location,
            rating_sleep_quality, rating_rooms, num_helpful_votes, via_mobile,
            sentiment_polarity, sentiment_subjectivity)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        review_rows,
    )
    conn.commit()


# ── Main pipeline ──────────────────────────────────────────────────────────────

def build_database(db_path: Path = DB_PATH, limit: int | None = 100_000) -> int:
    """
    Read review.json, filter to MIN_YEAR–MAX_YEAR, and write to SQLite.

    Args:
        db_path: Output database path.
        limit:   Max number of reviews to insert (None = all).

    Returns:
        Number of reviews inserted.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Remove existing DB so we start fresh
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    _create_schema(conn)
    _drop_indexes(conn)  # faster bulk inserts without indexes

    hotels_batch: dict[int, bool] = {}
    authors_batch: dict[str, dict] = {}
    reviews_batch: list[dict] = []
    total = 0

    t0 = time.time()
    print(f"[data_processing] Reading {REVIEW_JSON_PATH} …")
    with open(REVIEW_JSON_PATH, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            raw = json.loads(line)
            parsed = _parse_review(raw)
            if parsed is None:
                continue

            hotels_batch[parsed["hotel_id"]] = True
            aid = parsed["author"]["author_id"]
            if aid and aid not in authors_batch:
                authors_batch[aid] = parsed["author"]
            reviews_batch.append(parsed["review"])
            total += 1

            if len(reviews_batch) >= BATCH_SIZE:
                _insert_batch(conn, hotels_batch, authors_batch, reviews_batch)
                hotels_batch.clear()
                authors_batch.clear()
                reviews_batch.clear()
                print(f"  … {total:,} reviews inserted ({line_no:,} lines scanned)")

            if limit and total >= limit:
                break

    # flush remaining
    if reviews_batch:
        _insert_batch(conn, hotels_batch, authors_batch, reviews_batch)

    print(f"[data_processing] Creating indexes …")
    _create_indexes(conn)

    # Update hotel review counts
    conn.execute("""
        UPDATE hotels SET num_reviews = (
            SELECT COUNT(*) FROM reviews WHERE reviews.hotel_id = hotels.hotel_id
        )
    """)
    conn.commit()

    elapsed = time.time() - t0
    print(f"[data_processing] Done — {total:,} reviews in {elapsed:.1f}s → {db_path}")
    conn.close()
    return total


def build_sample_database(
    source_db: Path = DB_PATH,
    sample_db: Path = SAMPLE_DB_PATH,
    sample_size: int = SAMPLE_SIZE,
) -> int:
    """
    Create a small sample DB from the full DB for TA testing.
    Samples `sample_size` reviews randomly.
    """
    if sample_db.exists():
        sample_db.unlink()

    src = sqlite3.connect(str(source_db))
    dst = sqlite3.connect(str(sample_db))

    # Create schema in destination
    _create_schema(dst)

    # Sample review IDs
    ids = [
        row[0]
        for row in src.execute("SELECT review_id FROM reviews").fetchall()
    ]
    sampled_ids = random.sample(ids, min(sample_size, len(ids)))
    placeholders = ",".join("?" * len(sampled_ids))

    # Copy sampled reviews
    rows = src.execute(
        f"SELECT * FROM reviews WHERE review_id IN ({placeholders})", sampled_ids
    ).fetchall()
    cols = [d[0] for d in src.execute("SELECT * FROM reviews LIMIT 1").description]
    insert_sql = f"INSERT OR IGNORE INTO reviews ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
    dst.executemany(insert_sql, rows)

    # Copy related hotels
    hotel_ids = list({r[1] for r in rows})  # hotel_id is col index 1
    ph2 = ",".join("?" * len(hotel_ids))
    hotel_rows = src.execute(
        f"SELECT * FROM hotels WHERE hotel_id IN ({ph2})", hotel_ids
    ).fetchall()
    dst.executemany(
        "INSERT OR IGNORE INTO hotels (hotel_id, num_reviews) VALUES (?, ?)",
        hotel_rows,
    )

    # Copy related authors
    author_ids = list({r[2] for r in rows if r[2]})  # author_id is col index 2
    ph3 = ",".join("?" * len(author_ids))
    author_rows = src.execute(
        f"SELECT * FROM authors WHERE author_id IN ({ph3})", author_ids
    ).fetchall()
    author_cols = [d[0] for d in src.execute("SELECT * FROM authors LIMIT 1").description]
    dst.executemany(
        f"INSERT OR IGNORE INTO authors ({','.join(author_cols)}) VALUES ({','.join('?' * len(author_cols))})",
        author_rows,
    )

    # Update hotel counts in sample
    dst.execute("""
        UPDATE hotels SET num_reviews = (
            SELECT COUNT(*) FROM reviews WHERE reviews.hotel_id = hotels.hotel_id
        )
    """)
    dst.commit()

    count = dst.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    print(f"[data_processing] Sample DB → {count:,} reviews → {sample_db}")
    src.close()
    dst.close()
    return count


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_only = "--sample" in sys.argv

    if not sample_only:
        build_database()

    build_sample_database()
    print("[data_processing] All done ✓")
