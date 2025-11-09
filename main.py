import os
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import db, create_document, get_documents
from schemas import Document

app = FastAPI(title="Enterprise PDF Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Utility converters
class MongoJSONEncoder:
    @staticmethod
    def encode(doc: Dict[str, Any]):
        if not doc:
            return doc
        out = {}
        for k, v in doc.items():
            # Lazy import to avoid startup failure if bson isn't available yet
            try:
                from bson import ObjectId  # type: ignore
                if isinstance(v, ObjectId):
                    out[k] = str(v)
                    continue
            except Exception:
                pass
            out[k] = v
        return out


@app.get("/health")
@app.get("/")
def read_root():
    return {"message": "Enterprise PDF Intelligence Backend", "db": "ok" if db is not None else "unavailable"}


@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")

    inserted = 0
    for f in files:
        # Minimal ingestion: store basic metadata and create naive chunks
        content = await f.read()
        text = content.decode("utf-8", errors="ignore") if content else ""
        title = f.filename

        doc = Document(title=title, pages=None, table_count=0, metadata={"size": len(content)})
        doc_id = create_document("document", doc)

        # naive split into chunks of ~500 chars
        step = 500
        chunks = [text[i:i+step] for i in range(0, len(text), step)] or [""]
        for idx, ch in enumerate(chunks):
            # simple sparse embedding: term frequency dict for demonstration
            terms = [t for t in ch.lower().split() if t.isalpha() or t.isalnum()]
            emb: Dict[str, float] = {}
            for t in terms:
                emb[t] = emb.get(t, 0.0) + 1.0
            chunk_doc = {
                "doc_id": doc_id,
                "text": ch,
                "page": None,
                "embedding": emb,
                "metadata": {"title": title, "chunk": idx},
            }
            db["chunk"].insert_one(chunk_doc)
        inserted += 1

    return {"inserted": inserted}


@app.get("/documents")
def list_documents():
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    docs = get_documents("document")
    return [MongoJSONEncoder.encode(d) for d in docs]


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    try:
        from bson import ObjectId  # type: ignore
        _id = ObjectId(doc_id)
        res = db["document"].delete_one({"_id": _id})
    except Exception:
        # Fallback: attempt delete by string id if conversion fails
        res = db["document"].delete_one({"_id": doc_id})

    db["chunk"].delete_many({"doc_id": doc_id})

    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Not found")
    return {"deleted": True}


class Answer(BaseModel):
    text: str


@app.get("/search")
def semantic_search(query: str = Query(..., min_length=1), top_k: int = 5):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    # Simple sparse retrieval: cosine-like on TF vectors
    import math

    def score(q_vec: Dict[str, float], d_vec: Dict[str, float]):
        if not q_vec or not d_vec:
            return 0.0
        dot = sum(q_vec.get(t, 0.0) * d_vec.get(t, 0.0) for t in q_vec.keys())
        qn = math.sqrt(sum(v*v for v in q_vec.values()))
        dn = math.sqrt(sum(v*v for v in d_vec.values()))
        return dot / (qn * dn) if qn and dn else 0.0

    # build query vector
    tokens = [t for t in query.lower().split() if t.isalpha() or t.isalnum()]
    q_vec: Dict[str, float] = {}
    for t in tokens:
        q_vec[t] = q_vec.get(t, 0.0) + 1.0

    matches = []
    for ch in db["chunk"].find({}).limit(2000):
        s = score(q_vec, ch.get("embedding", {}))
        if s > 0:
            matches.append({
                "doc_id": ch.get("doc_id"),
                "text": ch.get("text", "")[:500],
                "score": float(s),
                "metadata": ch.get("metadata", {}),
            })
    matches.sort(key=lambda x: x["score"], reverse=True)
    matches = matches[:top_k]

    # naive answer: concatenate top chunks
    answer_text = " ".join(m["text"] for m in matches)[:800]
    return {"matches": matches, "answer": Answer(text=answer_text)}


@app.get("/test")
def test_database():
    ok = db is not None
    return {"backend": "running", "database": "connected" if ok else "not available"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
