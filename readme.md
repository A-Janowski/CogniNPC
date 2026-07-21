# CogniNPC

**Dynamic Personality Profiling and Interaction Generation for RPGs using Large Language Models**

CogniNPC is an experimental backend architecture designed to replace classic, static dialogue trees in video games with a fully dynamic, AI-driven system. This project utilizes local Small Language Models (SLMs) running in real-time, augmented by a Retrieval-Augmented Generation (RAG) architecture that serves as a long-term memory for Non-Player Characters (NPCs).

## 🎯 Project Goals & Academic Context
This repository contains the core API and logic for a Master of Science in Computer Science thesis project at the Cracow University of Technology (expected defense: September 2026). 

The research focuses on multi-agent system behavior within a gaming environment, specifically evaluating:
* **Dynamic Profiling:** Shifting NPC attitudes based on system prompts, psychological models (OCEAN), and player interactions.
* **Information Propagation (Gossip System):** Observing autonomous knowledge transfer and information distortion (the "telephone game" effect) between NPCs.
* **Performance Metrics:** Balancing "intelligence" with hardware constraints (VRAM, latency) on consumer-grade hardware.

## ⚙️ Architecture & Tech Stack
The system follows a strict Client-Server architecture, ensuring a clean separation between AI logic and graphical rendering.

* **Language & Framework:** Python 3.10, FastAPI (API Orchestrator)
* **Inference Engine:** Ollama (hosting quantized models like Llama-3-8B or Phi-3)
* **Vector Database (Memory / RAG):** ChromaDB (partitioned for individual NPC memory access)
* **Containerization:** Docker & Docker Compose

## 📁 Repository Structure
```text
CogniNPC/
├── app/
│   ├── main.py             # FastAPI application entry point
│   ├── api/                # API routes (endpoints.py)
│   ├── core/               # Configuration, logging, and dynamic prompt templates (prompt_builder.py)
│   ├── models/             # Pydantic schemas for data validation (schemas.py)
│   └── services/           # Business logic (chromadb_service.py, npc_service.py, ollama_service.py)
├── data/
│   └── npc_profiles/       # Static YAML configuration files for Fixed-Persona SLMs (e.g., smith_01.yaml)
├── Dockerfile              # Container definition for the backend
├── docker-compose.yaml     # Multi-container orchestration (API + Vector DB)
└── requirements.txt        # Python dependencies
```

## 🚀 Getting Started

### Prerequisites
* Docker and docker-compose installed.
* Local instance of Ollama running.

### Installation & Execution
1. Clone the repository.
2. Build and start the containerized environment:
   ```bash
   docker-compose up --build
   ```
3. The API will be available at `http://localhost:8000` (FastAPI Swagger UI at `/docs`).

## 👤 Author
**Adrian Janowski**  
*Backend & Cloud Software Engineer*  
*Cracow University of Technology*