# CogniNPC 🧠🤖

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black.svg)](https://ollama.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-orange.svg)](https://www.trychroma.com/)

**CogniNPC** is an advanced backend system for creating, managing, and interacting with intelligent Non-Player Characters (NPCs). The system combines the power of local Large Language Models (LLMs) with vector-based semantic memory, enabling deep and coherent conversations.

The project is designed for easy integration with game engines (such as Unity) or external applications via an efficient REST API.

---

## ✨ Key Features

*   **Local LLMs (Ollama):** Dialogue generation runs entirely locally using the Ollama engine, ensuring privacy and eliminating external API costs.
*   **Semantic Memory (ChromaDB):** A built-in vector database allows NPCs to maintain context and "remember" past interactions (RAG architecture).
*   **Profile Management:** Character traits, personalities, and knowledge bases (e.g., Aldric, Elara, Garret) are easily configurable via YAML files.
*   **Efficient REST API:** Built on the FastAPI framework, providing asynchronous processing and auto-generated documentation (Swagger UI).
*   **Easy Deployment:** The project is fully containerized using Docker and Docker Compose for seamless setup.
*   **Evaluation Module:** The repository includes testing tools (Jupyter Notebooks) for NPC psychometric verification (OCEAN/Big Five model) and performance benchmarking.

---

## 📂 Repository Structure

```text
CogniNPC/
├── app/
│   ├── api/                    # REST API endpoint definitions (endpoints.py)
│   ├── core/                   # Configuration, loggers, prompt templates (prompt_builder.py)
│   ├── models/                 # Pydantic data models (schemas.py)
│   ├── services/               # Business logic: 
│   │   ├── chromadb_service.py # Vector database integration
│   │   ├── npc_service.py      # Character management
│   │   └── ollama_service.py   # Communication with LLMs
│   └── main.py                 # Main FastAPI application entry point
├── data/
│   └── npc_profiles/                  # Character profiles saved in YAML format (e.g., thorin_01.yaml)
├── tests/                             # Testing and analytical scripts
│   ├── Ocean_Validation.ipynb         # Psychometric character validation
│   ├── Performance_Benchmark_v2.ipynb # System performance tests
│   └── results_test2/                 # Generation and evaluation results from tests
├── Dockerfile                         # Backend application image definition
├── docker-compose.yaml                # Container orchestration (Backend, ChromaDB, Ollama)
├── requirements.txt                   # Python library dependencies
└── readme.md                          # Project documentation
```

---

## 🚀 Quick Start

### Prerequisites
*   [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/) installed (for the recommended Docker method).
*   (Optional) [Ollama](https://ollama.com/) (for local execution without Docker).
*   (Optional) Python 3.10+ (for local execution without Docker).

### Running via Docker Compose (Recommended)
With this method, you **do not** need to install Ollama on your host machine. Docker will run it inside a container.

1. Clone the repository:
   ```bash
   git clone https://github.com/a-janowski/cogninpc.git
   cd cogninpc
   ```

2. Start the environment:
   ```bash
   docker-compose up -d --build
   ```

3. **Download the LLM model** inside the Ollama container. Wait for the containers to fully start, then run:
   ```bash
   docker-compose exec ollama-service ollama pull llama3.1:8b
   ```

4. The application will be available at: `http://localhost:8000`
5. Interactive API documentation (Swagger UI): `http://localhost:8000/docs`

### Local Execution (Development)
If you choose not to use Docker, you must install Ollama manually on your operating system.

1. Download and install **Ollama** from [ollama.com](https://ollama.com/).
2. Pull your preferred model via your local terminal:
   ```bash
   ollama pull llama3.1:8b
   ```
3. Install Python dependencies for the backend:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the development server:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

---


## 🛠️ Creating New Characters (NPCs)

To add a new NPC to the system, simply create a new `.yaml` file in the `data/npc_profiles/` directory.

Example file structure (e.g., `thorin_01.yaml`):
```yaml
id: thorin_01
name: Thorin Ironweaver
profession: Blacksmith
backstory: >
  A master blacksmith who has spent his entire life working at the forge. You are gruff, 
  straight to the point, and easily annoyed by those who waste your time with small talk. 
  You respect hard work and quality steel, and you only open up to those who prove their practical worth.
quirks:
  - "Uses short, clipped sentences interrupted by grunts or heavy sighs."
  - "Refers to fragile or talkative people as 'soft hands'."
ocean:
  openness: 20
  conscientiousness: 85
  extroversion: 20
  agreeableness: 15
  neuroticism: 40
```
The system will automatically load new profiles upon startup or when triggered via a dedicated refresh endpoint.

---

## 🧪 Validation and Tests

The project places a strong emphasis on NPC character consistency. The `tests/` folder contains Jupyter Notebooks, including:
* **Ocean_Validation.ipynb:** Analyzes conversation logs based on the Big Five psychological model (OCEAN) to verify if the generated responses align with the defined YAML profile.
* **Performance_Benchmark_v2.ipynb:** Analyzes query latency to Ollama and search times in the ChromaDB database.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Adrian Janowski**  
*Backend & Cloud Software Engineer*  
*CS Student at Cracow University of Technology*
