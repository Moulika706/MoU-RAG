import json
import re
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from fastembed import TextEmbedding

FILES = Path(__file__).parent / "files"
STORE = Path(__file__).parent / "chroma"

# Metadata for MOUs

DEPARTMENTS = {
    "cse": ["Computer Science and Engineering"],
    "ece": ["Electronics and Communication Engineering"],
    "it": ["Information Technology"],
    "ai": ["CSE–AIML & IoT", "CSE–AI & Data Science", "Robotics and AI"],
    "ml": ["CSE–AIML & IoT", "CSE–AI & Data Science"],
    "iot": ["CSE–AIML & IoT", "Internet of Things (IoT)"],
    "vlsi": ["Electronics Engineering (VLSI Design & Technology)"],
    "mech": ["Mechanical Engineering"],
    "auto": ["Automobile Engineering"],
    "aero": ["Aerospace Engineering"],
    "cyber": ["CSE–Cyber Security"],
    "civil": ["Civil Engineering"],
    "biotech": ["Biotechnology"],
}

DOMAINS = {
    "abroad": ["International Academic Collaboration", "Graduate Pathways"],
    "scholarship": ["Graduate Admissions"],
    "startup": ["Startup Incubation", "Entrepreneurship Development"],
    "research": ["Research and Development"],
    "internship": ["Internships & Placements"],
    "training": ["Industrial Training", "Skill Development"],
}

INTENTS = {
    "student": ["student", "looking for", "want to"],
    "faculty": ["faculty", "professor"],
    "startup": ["startup", "founder"],
    "abroad": ["abroad", "masters", "scholarship"],
}


class Query(BaseModel):
    query: str
    limit: int = 5


class Result(BaseModel):
    rank: int
    title: str
    partner: str
    score: float
    departments: list[str]
    domains: list[str]
    highlights: list[str]


class Response(BaseModel):
    query: str
    intent: str
    count: int
    results: list[Result]


mous = []
indexer = None


# Load MOU Documents
def load():
    data = []
    if not FILES.exists():
        return data
    
    skip = ["notifications.json", "users.json", "stars.json"]
    files = [f for f in FILES.glob("*.json") if f.name not in skip and not f.name.endswith(".meta.json")]
    
    for file in files:
        try:
            content = json.load(open(file, encoding="utf-8"))
            if content.get("is_active", True):
                data.append(normalize(content, file.stem))
        except:
            pass
    return data


# Normalize MOU Documents for Embedding
def normalize(data, fileid):
    highlights = data.get("key_highlights", [])
    return {
        "id": data.get("mou_id") or fileid,
        "title": data.get("title", "Untitled"),
        "partner": data.get("partner_name", "Unknown"),
        "summary": data.get("summary", ""),
        "highlights": highlights if isinstance(highlights, list) else [],
        "departments": data.get("departments", []),
        "domains": data.get("domains", []),
        "start": data.get("mou_start_date", ""),
        "end": data.get("mou_end_date", ""),
    }


# Combine MOU Document for Embedding
def combine(mou):
    parts = [mou["title"], mou["partner"], mou["summary"]]
    parts.extend(mou["highlights"])
    parts.extend(mou["departments"])
    parts.extend(mou["domains"])
    return " ".join(parts)


# Index MOU Documents for Search
class Indexer:
    def __init__(self, data):
        self.mous = {mou["id"]: mou for mou in data}
        self.embedder = TextEmbedding("BAAI/bge-base-en-v1.5")
        self.client = chromadb.PersistentClient(str(STORE))
        self.collection = self.client.get_or_create_collection("mous", metadata={"hnsw:space": "cosine"})
        
        if self.collection.count() == 0:
            self.build(data)
    
    def build(self, data):
        texts = [combine(mou) for mou in data]
        ids = [mou["id"] for mou in data]
        embeddings = list(self.embedder.embed(texts))
        self.collection.add(ids=ids, embeddings=[emb.tolist() for emb in embeddings])
    
    def search(self, query, limit=5):
        embedding = list(self.embedder.embed([query]))[0]
        results = self.collection.query(query_embeddings=[embedding.tolist()], n_results=limit * 2)
        return results["ids"][0], results["distances"][0]


# Process the Query for Metadata Extraction
def analyze(query):
    lower = query.lower()
    
    departments = set()
    for keyword, values in DEPARTMENTS.items():
        if keyword in lower:
            departments.update(values)
    
    domains = set()
    for keyword, values in DOMAINS.items():
        if keyword in lower:
            domains.update(values)
    
    intent = "general"
    for name, keywords in INTENTS.items():
        if any(kw in lower for kw in keywords):
            intent = name
            break
    
    return {"departments": list(departments), "domains": list(domains), "intent": intent}


# Calculate the Score for Each MOU (Basically this is just Ranking the MOUs)
def calculate(mou, analysis, distance):
    score = 1 - distance
    
    for dept in analysis["departments"]:
        if any(dept.lower() in d.lower() for d in mou["departments"]):
            score += 0.05
    
    for domain in analysis["domains"]:
        if any(domain.lower() in d.lower() for d in mou["domains"]):
            score += 0.03
    
    return min(score, 1.0)

# FastAPI Setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global mous, indexer
    print("Loading...")
    mous = load()
    print(f"Loaded {len(mous)} MOUs")
    print("Indexing...")
    indexer = Indexer(mous)
    print("Ready!")
    yield


app = FastAPI(title="MOURAG", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def root():
    return {"name": "MOURAG", "mous": len(mous)}


@app.get("/health")
def health():
    return {"ok": True, "mous": len(mous)}


@app.post("/search", response_model=Response)
def search(request: Query):
    if not indexer:
        raise HTTPException(500, "Not ready")
    if not request.query.strip():
        raise HTTPException(400, "Empty query")
    
    analysis = analyze(request.query)
    ids, distances = indexer.search(request.query, request.limit)
    
    results = []
    for i, mid in enumerate(ids):
        mou = indexer.mous.get(mid)
        if mou:
            score = calculate(mou, analysis, distances[i])
            results.append((mou, score))
    
    results.sort(key=lambda x: x[1], reverse=True)
    results = results[:request.limit]
    
    output = []
    for rank, (mou, score) in enumerate(results, 1):
        highlights = [re.sub(r'\*\*([^*]+)\*\*', r'\1', h)[:80] for h in mou["highlights"][:2]]
        output.append(Result(
            rank=rank,
            title=mou["title"],
            partner=mou["partner"],
            score=round(score, 3),
            departments=mou["departments"][:3],
            domains=mou["domains"][:3],
            highlights=highlights
        ))
    
    return Response(
        query=request.query,
        intent=analysis["intent"],
        count=len(output),
        results=output
    )


@app.get("/mous")
def list_mous():
    return [{"id": mou["id"], "title": mou["title"], "partner": mou["partner"]} for mou in mous]


@app.get("/mous/{mid}")
def get(mid: str):
    for mou in mous:
        if mou["id"] == mid:
            return mou
    raise HTTPException(404, "Not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
