# MOURAG: MoU Discovery Prototype

MOURAG is a prototype for searching Memorandums of Understanding (MoUs) through natural-language queries. It is designed for students and faculty who want to discover relevant collaborations for projects, startups, research, internships, training, or higher-education pathways.

The project uses a hybrid retrieval approach: semantic vector search for meaning, plus lightweight metadata extraction and scoring for departments, domains, and user intent. It uses FastEmbed locally, so no OpenAI API key is required.

## Data Privacy

Real college MoU data is not included in this repository because it may contain sensitive institutional information. The `files/` directory is ignored by Git except for a synthetic public sample file.

To run the project with private data locally:

1. Place your private `.json`, `.md`, and `.meta.json` MoU files inside `files/`.
2. Keep those files local. Do not commit them.
3. Delete the local `chroma/` directory whenever you replace the dataset so the vector store can be rebuilt.

## Hybrid RAG

1. **Semantic Vector Search**: The system converts MoU summaries and key highlights into high-dimensional embeddings using the BGE-base-en-v1.5 model. These are stored in a ChromaDB vector database to enable semantic similarity matching.
2. **Metadata Filtering and Scoring**: The system extracts entities such as departments (CSE, ECE, MECH), domains (AI, VLSI, Robotics), and user intent (Student, Faculty, Startup) from the natural-language query.

The final ranking is determined by a scoring algorithm that combines cosine similarity distance with metadata weights. This hybrid approach helps search results stay semantically similar and technically relevant to the user's departmental or domain constraints.

## Technical Stack

- **Backend**: FastAPI
- **Vector DB**: ChromaDB (Persistent)
- **Embedding Model**: FastEmbed (BAAI/bge-base-en-v1.5)
- **CLI / Demo App**: Python and Streamlit

## Installation and Execution

Install `uv` if it is not already available:

```bash
pip install uv
```

Install packages:

```bash
uv add -r requirements.txt
```

### Running the API Server

To start the FastAPI backend for web-based integration:

```bash
uv run python main.py
```

### Running the CLI

To use the command-line interface for rapid discovery:

```bash
uv run python app.py
```

### Running the Streamlit App

```bash
uv run streamlit run web.py
```

## Example

**Query:** I am a student looking for startup mentoring for a computer science project.

**System Analysis:**

- Intent: student
- Matched Departments: Computer Science and Engineering
- Matched Domains: Startup Incubation, Entrepreneurship Development

**Top Result:**

- Title: Sample MoU for Startup Mentoring
- Partner: Sample Innovation Labs
- Relevance Score: 0.73
- Justification: Match for student startup and project mentoring.
