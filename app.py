import json
import re
from pathlib import Path
import chromadb
from fastembed import TextEmbedding

FILES = Path(__file__).parent / "files"
STORE = Path(__file__).parent / "chroma"

# Keyword Mappings
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

# Load MOU Documents
def load():
    mous = []
    if not FILES.exists():
        return mous
    
    skip = ["notifications.json", "users.json", "stars.json"]
    files = [f for f in FILES.glob("*.json") if f.name not in skip and not f.name.endswith(".meta.json")]
    
    for file in files:
        try:
            data = json.load(open(file, encoding="utf-8"))
            if data.get("is_active", True):
                mous.append(normalize(data, file.stem))
        except:
            pass
    return mous

# Normalize MOU Documents
def normalize(data, fileid):
    highlights = data.get("key_highlights", [])
    return {
        "id": data.get("mou_id") or fileid,
        "title": data.get("title", "Untitled"),
        "partner": data.get("partner_name", "Unknown"),
        "type": data.get("mou_type", ""),
        "summary": data.get("summary", ""),
        "highlights": highlights if isinstance(highlights, list) else [],
        "departments": data.get("departments", []),
        "domains": data.get("domains", []),
        "tags": data.get("tags", []),
        "start": data.get("mou_start_date", ""),
        "end": data.get("mou_end_date", ""),
    }

# Combine MOU Document for Embed
def combine(mou):
    parts = [mou["title"], mou["partner"], mou["summary"]]
    parts.extend(mou["highlights"])
    parts.extend(mou["departments"])
    parts.extend(mou["domains"])
    return " ".join(parts)

# Index MOU Documents for Search
class Indexer:
    def __init__(self, mous):
        self.mous = {mou["id"]: mou for mou in mous}
        self.embedder = TextEmbedding("BAAI/bge-base-en-v1.5")
        self.client = chromadb.PersistentClient(str(STORE))
        self.collection = self.client.get_or_create_collection("mous", metadata={"hnsw:space": "cosine"})
        
        if self.collection.count() == 0:
            self.build(mous)
    
    def build(self, mous):
        texts = [combine(mou) for mou in mous]
        ids = [mou["id"] for mou in mous]
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

# Display the MOU
def display(rank, mou, score):
    lines = [
        f"MOU #{rank}: {mou['title']}",
        f"  Partner: {mou['partner']}",
        f"  Score: {score:.2f}",
        f"  Departments: {', '.join(mou['departments'][:3])}",
        f"  Domains: {', '.join(mou['domains'][:3])}",
    ]
    
    if mou["highlights"]:
        lines.append("  Highlights:")
        for highlight in mou["highlights"][:2]:
            clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', highlight)[:80]
            lines.append(f"    - {clean}")
    
    if mou["start"] or mou["end"]:
        lines.append(f"  Valid: {mou['start']} to {mou['end']}")
    
    return "\n".join(lines)


def main():
    print("\nMOURAG - MOU Discovery\n")
    
    print("Loading...")
    mous = load()
    print(f"Loaded {len(mous)} MOUs")
    
    print("Indexing...")
    indexer = Indexer(mous)
    print("Ready!\n")
    
    while True:
        try:
            query = input("Query: ").strip()
            if not query:
                continue
            if query in ["quit", "exit", "q"]:
                break
            
            analysis = analyze(query)
            ids, distances = indexer.search(query)
            
            results = []
            for i, mid in enumerate(ids):
                mou = indexer.mous.get(mid)
                if mou:
                    score = calculate(mou, analysis, distances[i])
                    results.append((mou, score))
            
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:5]
            
            print(f"\nFound {len(results)} MOUs:\n")
            for rank, (mou, score) in enumerate(results, 1):
                print(display(rank, mou, score))
                print()
            
            if analysis["intent"] != "general":
                print(f"Intent: {analysis['intent']}")
            if analysis["departments"]:
                print(f"Matched: {', '.join(analysis['departments'][:3])}")
            print()
            
        except KeyboardInterrupt:
            break
    
    print("Bye!")


if __name__ == "__main__":
    main()
