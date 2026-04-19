# -*- coding: utf-8 -*-
"""SQLite-backed FAQ semantic search server using FastAPI + ScaNN."""

import csv
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

import numpy as np
import scann
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# --- 1. CONFIGURATION ---
MODEL_NAME = "all-MiniLM-L6-v2"
K_NEIGHBORS = 1000
DB_PATH = os.getenv("FAQ_DB_PATH", "faq.db")
FAQ_CSV_PATH = os.getenv("FAQ_CSV_PATH", "problem_resolution.csv")

# Global State
embedding_model = None
searcher = None
loaded_train_embeddings = None
faq_rows: List[Dict[str, str]] = []


class SearchQuery(BaseModel):
    query_text: str
    k_neighbors: int = 5


class BenchmarkRequest(BaseModel):
    num_queries: int = 50


class FAQItem(BaseModel):
    question: str
    answer: str


class FAQBulkUpsertRequest(BaseModel):
    items: List[FAQItem]


class FAQCsvImportRequest(BaseModel):
    csv_path: str = FAQ_CSV_PATH
    reset_existing: bool = False


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def embedding_to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def blob_to_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def faq_text(question: str, answer: str) -> str:
    return f"Q: {question}\nA: {answer}"


def initialize_db() -> None:
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS faq_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                embedding BLOB,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(question, answer)
            )
            """
        )
        conn.commit()


def seed_faq_if_empty() -> None:
    seed_items = [
        FAQItem(
            question="How can I reset my account password?",
            answer="Use the password reset page and follow the verification steps sent to your email.",
        ),
        FAQItem(
            question="Why is my dashboard slow?",
            answer="Clear browser cache, disable heavy filters, and retry. If it persists, contact support.",
        ),
    ]

    with get_db_connection() as conn:
        row = conn.execute("SELECT COUNT(*) AS total FROM faq_items").fetchone()
        if row["total"] > 0:
            return

    upsert_faq_items(seed_items)


def upsert_faq_items(items: List[FAQItem]) -> None:
    texts = [faq_text(i.question.strip(), i.answer.strip()) for i in items]
    embeddings = embedding_model.encode(texts, convert_to_tensor=False)

    with get_db_connection() as conn:
        for item, emb in zip(items, embeddings):
            question = item.question.strip()
            answer = item.answer.strip()
            conn.execute(
                """
                INSERT INTO faq_items (question, answer, embedding, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(question, answer) DO UPDATE SET
                    embedding = excluded.embedding,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (question, answer, embedding_to_blob(emb)),
            )
        conn.commit()


def load_faq_rows() -> List[sqlite3.Row]:
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT id, question, answer, embedding FROM faq_items ORDER BY id"
        ).fetchall()
    return rows


def import_problem_resolution_csv(csv_path: str, reset_existing: bool = False) -> int:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    parsed_items: List[FAQItem] = []
    seen_pairs = set()

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"issue_description", "resolution_notes"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                "CSV must include columns: issue_description,resolution_notes"
            )

        for row in reader:
            question = (row.get("issue_description") or "").strip()
            answer = (row.get("resolution_notes") or "").strip()
            if not question or not answer:
                continue

            key = (question, answer)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            parsed_items.append(FAQItem(question=question, answer=answer))

    if not parsed_items:
        return 0

    texts = [faq_text(item.question, item.answer) for item in parsed_items]
    embeddings = embedding_model.encode(texts, convert_to_tensor=False)

    with get_db_connection() as conn:
        if reset_existing:
            conn.execute("DELETE FROM faq_items")

        for item, emb in zip(parsed_items, embeddings):
            conn.execute(
                """
                INSERT INTO faq_items (question, answer, embedding, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(question, answer) DO UPDATE SET
                    embedding = excluded.embedding,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (item.question, item.answer, embedding_to_blob(emb)),
            )

        conn.commit()

    return len(parsed_items)


def ensure_embeddings(rows: List[sqlite3.Row]) -> None:
    missing = [r for r in rows if r["embedding"] is None]
    if not missing:
        return

    texts = [faq_text(r["question"], r["answer"]) for r in missing]
    generated = embedding_model.encode(texts, convert_to_tensor=False)

    with get_db_connection() as conn:
        for row, emb in zip(missing, generated):
            conn.execute(
                "UPDATE faq_items SET embedding = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (embedding_to_blob(emb), row["id"]),
            )
        conn.commit()


def rebuild_index_from_db() -> None:
    global searcher, loaded_train_embeddings, faq_rows

    rows = load_faq_rows()
    if not rows:
        searcher = None
        loaded_train_embeddings = None
        faq_rows = []
        return

    ensure_embeddings(rows)
    rows = load_faq_rows()

    vectors = np.vstack([blob_to_embedding(r["embedding"]) for r in rows]).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    loaded_train_embeddings = vectors / norms

    faq_rows = [
        {
            "id": int(r["id"]),
            "question": r["question"],
            "answer": r["answer"],
        }
        for r in rows
    ]

    n_items = len(faq_rows)
    nn = min(K_NEIGHBORS, n_items)

    if n_items < 20:
        searcher = scann.scann_ops_pybind.builder(
            loaded_train_embeddings,
            nn,
            "dot_product",
        ).score_brute_force().build()
        return

    num_leaves = min(max(int(np.sqrt(n_items)), 10), n_items)
    builder = scann.scann_ops_pybind.builder(
        loaded_train_embeddings,
        nn,
        "dot_product",
    ).tree(
        num_leaves=num_leaves,
        num_leaves_to_search=max(int(num_leaves * 0.2), 2),
        training_sample_size=min(max(int(n_items * 0.2), 10), n_items),
    ).score_ah(
        2,
        anisotropic_quantization_threshold=0.2,
    ).reorder(
        nn * 5,
    )
    searcher = builder.build()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model

    print("--- SERVER STARTUP ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"1. Init model ({device})...")
    embedding_model = SentenceTransformer(MODEL_NAME, device=device)

    print(f"2. Init SQLite database at: {DB_PATH}")
    initialize_db()

    with get_db_connection() as conn:
        total = conn.execute("SELECT COUNT(*) AS total FROM faq_items").fetchone()["total"]

    if total == 0 and Path(FAQ_CSV_PATH).exists():
        print(f"3. Import FAQ records from CSV: {FAQ_CSV_PATH}")
        imported = import_problem_resolution_csv(FAQ_CSV_PATH, reset_existing=False)
        print(f"   Imported {imported} issue-resolution pairs")
    elif total == 0:
        print("3. CSV not found; seeding minimal fallback FAQs")
        seed_faq_if_empty()

    print("4. Build ScaNN index from FAQ table...")
    rebuild_index_from_db()
    print(f"--- READY ({len(faq_rows)} FAQ rows indexed) ---")

    yield
    print("--- SERVER SHUTDOWN ---")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok", "ready": searcher is not None, "faq_count": len(faq_rows)}


@app.get("/badge")
def get_badge():
    return {
        "schemaVersion": 1,
        "label": "status",
        "message": "online" if searcher is not None else "offline",
        "color": "green" if searcher is not None else "red",
    }


# @app.post("/faq/bulk_upsert")
# def faq_bulk_upsert(req: FAQBulkUpsertRequest):
#     if not req.items:
#         raise HTTPException(400, "items must not be empty")

#     upsert_faq_items(req.items)
#     rebuild_index_from_db()
#     return {"status": "ok", "faq_count": len(faq_rows)}


# @app.post("/faq/import_csv")
# def faq_import_csv(req: FAQCsvImportRequest):
#     try:
#         imported = import_problem_resolution_csv(
#             csv_path=req.csv_path,
#             reset_existing=req.reset_existing,
#         )
#         rebuild_index_from_db()
#         return {
#             "status": "ok",
#             "imported": imported,
#             "faq_count": len(faq_rows),
#             "csv_path": req.csv_path,
#         }
#     except FileNotFoundError as exc:
#         raise HTTPException(404, str(exc)) from exc
#     except ValueError as exc:
#         raise HTTPException(400, str(exc)) from exc


@app.post("/search")
def search(query: SearchQuery):
    if searcher is None:
        raise HTTPException(503, "Not ready. FAQ index is unavailable.")

    print("query: "+query.query_text +"; k = "+ str(query.k_neighbors))
    start = time.time()
    q_emb = embedding_model.encode([query.query_text], convert_to_tensor=False)[0]
    q_emb = q_emb / np.maximum(np.linalg.norm(q_emb), 1e-12)

    final_k = max(1, min(query.k_neighbors, len(faq_rows), K_NEIGHBORS))
    indices, distances = searcher.search(
        q_emb,
        final_num_neighbors=final_k,
        pre_reorder_num_neighbors=max(final_k * 5, K_NEIGHBORS * 5),
    )

    elapsed = (time.time() - start) * 1000

    results = []
    for rank, (idx, score) in enumerate(zip(indices, distances), start=1):
        row = faq_rows[int(idx)]
        results.append(
            {
                "rank": rank,
                "text": faq_text(row["question"], row["answer"]),
                "question": row["question"],
                "answer": row["answer"],
                "faq_id": row["id"],
                "similarity": float(score),
                "dataset_index": int(idx),
            }
        )

    return {"query": query.query_text, "results": results, "search_time_ms": elapsed}


# @app.post("/benchmark")
# def benchmark(req: BenchmarkRequest):
#     if searcher is None or loaded_train_embeddings is None:
#         raise HTTPException(503, "Not ready. FAQ index is unavailable.")

#     if not faq_rows:
#         raise HTTPException(400, "No FAQ data available for benchmark")

#     n_queries = max(1, min(req.num_queries, len(faq_rows)))
#     query_texts = [faq_text(item["question"], item["answer"]) for item in faq_rows[:n_queries]]

#     test_emb = embedding_model.encode(query_texts, convert_to_tensor=False)
#     test_emb = test_emb / np.maximum(np.linalg.norm(test_emb, axis=1, keepdims=True), 1e-12)

#     bf_start = time.time()
#     nn = min(K_NEIGHBORS, len(faq_rows))
#     bf_searcher = scann.scann_ops_pybind.builder(
#         loaded_train_embeddings,
#         nn,
#         "dot_product",
#     ).score_brute_force().build()
#     bf_idx, _ = bf_searcher.search_batched(test_emb)
#     bf_time = time.time() - bf_start

#     scann_start = time.time()
#     scann_idx, _ = searcher.search_batched(test_emb)
#     scann_time = time.time() - scann_start

#     k = min(5, nn)
#     recall_sum = 0.0
#     for i in range(n_queries):
#         recall_sum += len(set(bf_idx[i][:k]).intersection(set(scann_idx[i][:k]))) / k
#     avg_recall = recall_sum / n_queries

#     return {
#         "dataset_size": len(faq_rows),
#         "num_queries": n_queries,
#         "results": [
#             {
#                 "method": "Brute Force",
#                 "time_seconds": bf_time,
#                 "avg_ms_per_query": (bf_time / n_queries) * 1000,
#                 "recall": 1.0,
#             },
#             {
#                 "method": "ScaNN",
#                 "time_seconds": scann_time,
#                 "avg_ms_per_query": (scann_time / n_queries) * 1000,
#                 "recall": avg_recall,
#             },
#         ],
#     }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
