"""
daws/database/repository.py
-----------------------------
DAWS MongoDB Repository with strict JSON Schema validation.

Collections
-----------
recordings  — static 50-sample corpus (WER, H_k6, delta_pc1, u_asr per sample)
inferences  — real-time production runs (transcript → risk assessment log)

Usage
-----
    from daws.database import DAWSRepository
    repo = DAWSRepository()
    repo.log_inference({...})
    docs = repo.get_inferences(risk_level="red", limit=50)
"""

import logging
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)

# ── JSON Schema definitions ────────────────────────────────────────────────

_SCHEMA_RECORDINGS = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["filename", "wer", "u_asr", "created_at"],
        "additionalProperties": True,
        "properties": {
            "filename":   {"bsonType": "string"},
            "dialect":    {"bsonType": "string"},
            "gender":     {"bsonType": "string"},
            "age_range":  {"bsonType": "string"},
            "wer":        {"bsonType": "double", "minimum": 0.0, "maximum": 1.0},
            "u_asr":      {"bsonType": "double", "minimum": 0.0, "maximum": 1.0},
            "H_k6":       {"bsonType": "double", "minimum": 0.0},
            "delta_pc1":  {"bsonType": "double", "minimum": 0.0},
            "gt1":        {"bsonType": "string"},
            "w1":         {"bsonType": "string"},
            "created_at": {"bsonType": "date"},
        },
    }
}

_SCHEMA_INFERENCES = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["transcript", "u_asr", "u_llm", "u_pipeline", "risk_level", "created_at"],
        "additionalProperties": True,
        "properties": {
            "transcript":          {"bsonType": "string"},
            "u_asr":               {"bsonType": "double", "minimum": 0.0, "maximum": 1.0},
            "u_llm":               {"bsonType": "double", "minimum": 0.0},
            "u_pipeline":          {"bsonType": "double", "minimum": 0.0, "maximum": 1.0},
            "risk_level":          {"bsonType": "string", "enum": ["green", "yellow", "red"]},
            "clarifying_question": {"bsonType": ["string", "null"]},
            "created_at":          {"bsonType": "date"},
        },
    }
}


# ── Repository ─────────────────────────────────────────────────────────────

class DAWSRepository:
    """
    MongoDB repository for DAWS.  Connects lazily on first use.
    Falls back to no-op if MongoDB is unavailable (dashboard degrades gracefully).
    """

    DB_NAME   = "daws"
    MONGO_URI = "mongodb://localhost:27017"

    def __init__(self, uri: Optional[str] = None, db_name: Optional[str] = None):
        self._uri     = uri or self.MONGO_URI
        self._db_name = db_name or self.DB_NAME
        self._client  = None
        self._db      = None
        self._available = None  # None = not yet tested

    # ── internal ──────────────────────────────────────────────────────────

    def _connect(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            from pymongo import MongoClient, ASCENDING, DESCENDING
            from pymongo.errors import ServerSelectionTimeoutError

            self._client = MongoClient(self._uri, serverSelectionTimeoutMS=3_000)
            self._client.admin.command("ping")  # raises if unreachable
            self._db = self._client[self._db_name]
            self._setup_collections()
            self._available = True
            log.info(f"MongoDB connected: {self._uri}/{self._db_name}")
        except Exception as exc:
            self._available = False
            log.warning(f"MongoDB unavailable ({exc}) — repository is no-op.")
        return self._available

    def _setup_collections(self):
        from pymongo import ASCENDING, DESCENDING

        existing = self._db.list_collection_names()

        if "recordings" not in existing:
            self._db.create_collection(
                "recordings",
                validator=_SCHEMA_RECORDINGS,
                validationAction="warn",
            )
        col_rec = self._db["recordings"]
        col_rec.create_index([(("filename", ASCENDING))], unique=True, background=True)
        col_rec.create_index([("dialect",  ASCENDING)], background=True)
        col_rec.create_index([("wer",      DESCENDING)], background=True)

        if "inferences" not in existing:
            self._db.create_collection(
                "inferences",
                validator=_SCHEMA_INFERENCES,
                validationAction="warn",
            )
        col_inf = self._db["inferences"]
        col_inf.create_index([("risk_level",  ASCENDING)],  background=True)
        col_inf.create_index([("created_at",  DESCENDING)], background=True)

    def _col(self, name: str):
        return self._db[name] if self._available else None

    # ── recordings collection ─────────────────────────────────────────────

    def log_recording(self, doc: dict) -> Optional[str]:
        """
        Upsert a recording from the static corpus.

        Required fields: filename, wer, u_asr.
        Optional: dialect, gender, age_range, H_k6, delta_pc1, gt1, w1.
        """
        if not self._connect():
            return None
        try:
            doc = {**doc, "created_at": doc.get("created_at", datetime.now(timezone.utc))}
            result = self._col("recordings").update_one(
                {"filename": doc["filename"]},
                {"$set": doc},
                upsert=True,
            )
            return str(result.upserted_id or doc["filename"])
        except Exception as exc:
            log.error(f"log_recording failed: {exc}")
            return None

    def get_recordings(self, limit: int = 200) -> list[dict]:
        """Fetch recordings sorted by WER descending."""
        if not self._connect():
            return []
        try:
            from pymongo import DESCENDING
            cursor = self._col("recordings").find(
                {}, {"_id": 0}
            ).sort("wer", DESCENDING).limit(limit)
            return list(cursor)
        except Exception as exc:
            log.error(f"get_recordings failed: {exc}")
            return []

    # ── inferences collection ─────────────────────────────────────────────

    def log_inference(self, doc: dict) -> Optional[str]:
        """
        Insert a real-time inference record.

        Required fields: transcript, u_asr, u_llm, u_pipeline, risk_level.
        Optional: audio_path, llm_response, clarifying_question.
        """
        if not self._connect():
            return None
        try:
            doc = {**doc, "created_at": doc.get("created_at", datetime.now(timezone.utc))}
            result = self._col("inferences").insert_one(doc)
            return str(result.inserted_id)
        except Exception as exc:
            log.error(f"log_inference failed: {exc}")
            return None

    def get_inferences(
        self,
        risk_level: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Fetch inference records, newest first.
        Optionally filter by risk_level ('green'|'yellow'|'red').
        """
        if not self._connect():
            return []
        try:
            from pymongo import DESCENDING
            filt = {}
            if risk_level:
                filt["risk_level"] = risk_level.lower()
            cursor = self._col("inferences").find(
                filt, {"_id": 0}
            ).sort("created_at", DESCENDING).limit(limit)
            return list(cursor)
        except Exception as exc:
            log.error(f"get_inferences failed: {exc}")
            return []

    def get_risk_counts(self) -> dict[str, int]:
        """Return {risk_level: count} for all inferences."""
        if not self._connect():
            return {}
        try:
            pipeline = [{"$group": {"_id": "$risk_level", "n": {"$sum": 1}}}]
            return {
                doc["_id"]: doc["n"]
                for doc in self._col("inferences").aggregate(pipeline)
                if doc["_id"]
            }
        except Exception as exc:
            log.error(f"get_risk_counts failed: {exc}")
            return {}
