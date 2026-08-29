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

*   **Local LLM inference (Ollama):** Dialogue generation runs entirely on the host machine,
    ensuring privacy, reproducibility, and zero external API cost.
*   **Semantic memory (ChromaDB):** A vector store lets each NPC retrieve relevant past
    interactions and world knowledge (RAG architecture). Memories are partitioned per NPC
    through metadata filtering on a single collection.
*   **Explicit personality model:** Character behaviour is driven by a five-trait OCEAN
    (Big Five) profile mapped onto prompt directives, with a Valence–Arousal affective
    state layered on top.
*   **Gossip subsystem:** NPCs can exchange memories with one another, allowing information
    to propagate — and degrade — across the character population.
*   **REST API (FastAPI):** Pydantic request/response validation and auto-generated
    OpenAPI documentation (Swagger UI).
*   **Containerised deployment:** Backend, ChromaDB, and Ollama are orchestrated with
    Docker Compose.
*   **Evaluation tooling:** Jupyter notebooks for latency/throughput benchmarking and for
    evaluating behavioural consistency against the authored OCEAN profile.

---

## 🖥️ Hardware Requirements
 
Inference is the bottleneck of this system, and running it on CPU makes it effectively
unusable (replies in the tens of seconds). A CUDA-capable GPU is strongly recommended.
 
| Component | Minimum | Reference setup |
|---|---|---|
| GPU | 8 GB VRAM | NVIDIA RTX 5070 Ti, 16 GB VRAM |
| RAM | 16 GB | 32 GB |
| Disk | ~10 GB for the model weights | — |
| OS | Linux, or Windows with WSL2 | Windows 11 + WSL2 |
 
For GPU access inside Docker you also need the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
installed on the host. Verify with:
 
```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```
 
If this fails, Ollama will silently fall back to CPU.
 
---

## 📂 Repository Structure

```text
CogniNPC/
├── app/
│   ├── api/                    # REST API endpoint definitions (endpoints.py)
│   ├── core/                   # Configuration, loggers, prompt templates (prompt_builder.py)
│   ├── models/                 # Pydantic data models (schemas.py)
│   ├── services/               
│   │   ├── chromadb_service.py # Vector database integration
│   │   ├── npc_service.py      # Character management
│   │   └── ollama_service.py   # Communication with LLMs
│   └── main.py                 # Main FastAPI application entry point
├── data/
│   └── npc_profiles/                  # Character profiles saved in YAML format (e.g., thorin_01.yaml)
├── tests/                             # Testing and analytical scripts
│   ├── Ocean_Validation.ipynb         # Psychometric character validation
│   ├── Performance_Benchmark_v2.ipynb # System performance tests (Latency, throughput, VRAM occupancy)
│   └── results_test2/                 # Generation and evaluation results from tests
├── Dockerfile                         # Backend image definition
├── docker-compose.yaml                # Container orchestration (Backend, ChromaDB, Ollama)
├── requirements.txt                   
└── readme.md                          
```

**Note on the Unity client:** the game client is deliberately *not* part of the Compose
stack. Unity is a graphical, host-bound application that requires GPU display access and
its own editor tooling; containerising it would add complexity without benefit, since it
communicates with the backend over HTTP like any other external client.

---

## 🚀 Quick Start

### Prerequisites
*   [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/)
*   NVIDIA Container Toolkit (see *Hardware Requirements* above)
*   *(Only for non-Docker development)* Python 3.10+ and a local [Ollama](https://ollama.com/) install

### Running via Docker Compose (Recommended)
With this method you do **not** need Ollama installed on the host — it runs in a container.

1. Clone the repository:
   ```bash
   git clone https://github.com/a-janowski/cogninpc.git
   cd cogninpc
   ```

2. Start the environment:
   ```bash
   docker-compose up -d --build
   ```

3. Pull the LLM inside the Ollama container (wait for the containers to be healthy first):
   ```bash
   docker-compose exec ollama-service ollama pull llama3.1:latest
   ```

4. The API is available at `http://localhost:8000`
5. Interactive documentation (Swagger UI): `http://localhost:8000/docs`

> The `llama3.1:latest` tag is used consistently throughout the project, including in all
> benchmark runs. Using a different tag will produce results that are not comparable with
> the published measurements.

### Local Execution (Development)
If you choose not to use Docker, you must install Ollama manually on your operating system.

1. Download and install **Ollama** from [ollama.com](https://ollama.com/).
2. Pull your preferred model via your local terminal:
   ```bash
   ollama pull llama3.1:latest
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

## 🎮 Integration Guide — Using CogniNPC in Your Own Game
 
### 1. Minimal integration
 
Start the backend, place a profile in `data/npc_profiles/`, and POST the player's
utterance together with the NPC identifier.
 
```http
POST /api/chat
Content-Type: application/json
 
{
  "npc_id": "thorin_01",
  "player_message": "Do you sell steel here?"
}
```
 
Response:
 
```json
{
  "npc_id": "thorin_01",
  "response": "Aye. Steel, iron... whatever your coin buys.",
  "memory_used": true,
  "metrics": {
    ...
  }
}
```
 
### 2. What the backend handles for you
 
The client sends one message and receives one reply. Everything between those two points
is server-side:
 
- retrieval of relevant memories for that NPC from the vector store,
- injection of the OCEAN traits and current affective state as behavioural directives
  in the system prompt,
- assembly of the full prompt (persona, retrieved context, dialogue turn),
- persistence of the exchange back into the NPC's memory.
The client is therefore effectively stateless — it only needs to track `npc_id`
 
### 3. Unity example (`UnityWebRequest`)
 
```csharp
[Serializable] public class ChatRequest  { public string npc_id, player_message; }
[Serializable] public class ChatResponse { public string npc_id, response; public bool memory_used;}

    public async Task<ChatResponse> SendChatMessageAsync(string npcId, string message)
    {
        var url = "http://localhost:8000/api/chat";
        var requestData = new ChatRequest
        {
            npc_id = npcId,
            player_message = message
        };
        var jsonBody = JsonUtility.ToJson(requestData);
        var rawBody = Encoding.UTF8.GetBytes(jsonBody);

        using (var request = new UnityWebRequest(url, "POST"))
        {
            request.uploadHandler = new UploadHandlerRaw(rawBody);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            
            var operation = request.SendWebRequest();
            while (!operation.isDone)
            {
                await Task.Yield();
            }

            if (request.result == UnityWebRequest.Result.Success)
            {
                var jsonResponse = request.downloadHandler.text;
                return JsonUtility.FromJson<ChatResponseDTO>(jsonResponse);
            }
            
            Debug.LogError($"CogniNPC API Error: {request.error} | Response: {request.downloadHandler.text}");
            return new ChatResponse
            { 
                response_text = "[Your interlocutor fell asleep and is unresponsive.]", 
                memory_used = false 
            };
        }
    }
```
 
### 4. Practical notes before you build on this
 
These are the things that most often make a first integration look broken when it is not:
 
- **Latency is measured in seconds, not milliseconds.** Expect roughly 1–3 s per reply on
  a consumer GPU. Design the interaction around it: a "thinking" animation, an idle
  barks system, or a typewriter reveal all work far better than a frozen dialogue box.
- **Requests are served one at a time.** Ollama defaults to `OLLAMA_NUM_PARALLEL=1`, so
  concurrent requests queue on the inference server rather than running in parallel.
  Either queue NPC requests client-side or raise that limit, budgeting VRAM accordingly.
- **The backend endpoints are synchronous by design.** They are defined with `def`, not
  `async def`, so FastAPI dispatches them to a thread pool. This is intentional: blocking
  inference calls inside `async def` handlers serialise through the event loop and
  severely inflate latency under concurrency.
- **Author your content in English.** Profiles, prompts, and world content are written in
  English because Llama 3.1 8B degrades noticeably in other languages. This is a property
  of the model, not of the system.
- **Cold start is slow.** The first request after container startup includes model load
  time. Send a warm-up request before the player can reach an NPC.
  
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

Each OCEAN trait is an integer in the range 0–100 and is translated into an explicit
behavioural directive in the system prompt. Values are binned into bands; the band
boundaries are project-specific design choices rather than psychometric norms.
 
Profiles are loaded at startup, or on demand through the profile-refresh endpoint.
 
**Bundled characters:** Thorin Ironweaver, Elara Vance, Garret Strongstew, Keldorn Vance, Aldric—
five profiles spanning distinct trait configurations. `aldric_neutral.json` is a neutral
carrier profile used as a control condition in the OCEAN ablation study; it is not
intended for gameplay.

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
