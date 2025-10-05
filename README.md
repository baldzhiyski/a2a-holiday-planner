# 🧭 A2A Trip Planner (OpenAI Edition)

This repository demonstrates a **multi-agent orchestration system using the A2A protocol**, powered entirely by **OpenAI models** and combining **CrewAI** and **LangGraph**.

It’s a complete, runnable example of distributed LLM agents collaborating over HTTP to plan a full trip — flights, hotels, activities, and budget allocation — and then compose and book itineraries automatically.

---

## 🚀 Overview

### ✳️ What This Demo Shows
This project demonstrates:
- The **A2A (Agent-to-Agent)** protocol for inter-agent communication.
- A **LangGraph host agent** coordinating several **CrewAI and LangGraph micro-agents**.
- Multi-step, structured, and reasoning-heavy LLM workflows with **strict JSON schema outputs**.
- An **OpenAI-only** implementation — all LLM calls use OpenAI models (no Google ADK, no Gemini).

### 🧩 Architecture
```
a2a-tripplanner-openai/
├─ pyproject.toml
├─ README.md
├─ host/
│  ├─ main.py
│  └─ app/
│     ├─ host_graph.py
│     ├─ agent_executor.py
│     ├─ remote_connection.py
│     ├─ itinerary_tools.py
│     └─ schemas.py
├─ flights_crewai/
│  ├─ main.py
│  └─ app/
│     ├─ agent.py
│     └─ agent_executor.py
├─ hotels_crewai/
│  ├─ main.py
│  └─ app/
│     ├─ agent.py
│     └─ agent_executor.py
├─ activities_crewai/
│  ├─ main.py
│  └─ app/
│     ├─ agent.py
│     └─ agent_executor.py
└─ budget_langgraph/
   ├─ main.py
   └─ app/
      ├─ agent.py
      └─ agent_executor.py
```

---

## 🧠 Agents Overview

| Agent | Tech | Role | Description |
|-------|------|------|--------------|
| **TripPlannerHost** | LangGraph | Coordinator | Orchestrates the trip planning flow; sends messages to other agents via A2A; merges results and creates itineraries. |
| **FlightsAgent** | CrewAI | Data fetcher | Uses CrewAI built-in tools (`SerperDevTool`, `ScrapeWebsiteTool`) to find flight options, then outputs structured JSON results. |
| **HotelsAgent** | CrewAI | Data fetcher | Searches and scrapes hotel listings, normalizes to JSON (`HotelItem` schema). |
| **ActivitiesAgent** | CrewAI | Data fetcher | Scrapes things to do (food tours, museums, etc.) within a given date range. |
| **BudgetAgent** | LangGraph | Reasoner | Computes budget splits and constraints (flight, hotel, activities) given total budget and preferences. |

---

## ⚙️ Technologies Used
| Stack | Purpose |
|--------|----------|
| **A2A Protocol** | Enables peer-to-peer HTTP communication between LLM agents. |
| **LangGraph** | Orchestrates reasoning and structured output for the Host & Budget agents. |
| **CrewAI** | Simplifies tool-based scraping, searching, and content extraction for Flights/Hotels/Activities. |
| **OpenAI GPT-4o-mini** | Core reasoning and text generation model for all agents. |
| **Uvicorn + Starlette (A2A App)** | Provides a lightweight async HTTP interface for each agent. |
| **Pydantic** | Defines schemas for all structured LLM outputs (e.g., `Flight`, `Hotel`, `Activity`, `Itinerary`). |
| **uv** | Used as the lightweight package/runtime manager for running each agent. |

---

## 🪄 How It Works (Flow Summary)

1. **You send a prompt** to the Host (e.g. “Plan a 4-day trip from Berlin to Lisbon for 2 people, €2200 budget.”)
2. The **TripPlannerHost** extracts user intent (origin, destination, budget, dates, preferences).
3. It calls `send_message()` to query each micro-agent via A2A:
   - `BudgetAgent` → returns budget caps (e.g. flights €900, hotel €800, activities €500)
   - `FlightsAgent` → returns a JSON list of flight options
   - `HotelsAgent` → returns a JSON list of hotel options
   - `ActivitiesAgent` → returns a JSON list of local activities
4. Host calls `compose_itineraries()`:
   - Merges results
   - Picks 2–3 best itineraries based on timing, price, and preferences
   - Returns a summary + a full structured JSON artifact
5. You can respond: **“Book option 2”**, and the Host calls `book_itinerary()` to simulate bookings.

---

## 🧾 Example Flow

### Input
```
Plan a 4-day trip to Lisbon from Berlin, 
departing 2025-11-10, returning 2025-11-14.
Budget €2200. Prefer boutique hotels and food activities.
```

### Behind the Scenes
```
Host → BudgetAgent: compute_caps(...)
Host → FlightsAgent: search flights
Host → HotelsAgent: scrape hotels
Host → ActivitiesAgent: find tours and attractions
Host → Host: compose_itineraries(...)
```

### Output (Artifact)
```json
{
  "candidates": [
    {
      "summary": "Berlin → Lisbon, Nov 10–14, 2 people, total €2170",
      "price_breakdown_eur": {"flights": 950, "hotel": 880, "activities": 340},
      "score": 0.94,
      "hotel": {"name": "Lisbon Boutique Stay", "rating": 4.7},
      "days": [
        {"date_iso": "2025-11-11", "morning": "Pastel de Nata class", "afternoon": "Alfama walking tour"}
      ]
    }
  ]
}
```

### Booking Command
```
Book option 1
```
→ Host returns a mock confirmation with flight/hotel booking IDs.

---

## 🧰 Installation

```bash
# Clone this repo
git clone https://github.com/baldzhiyski/a2a-holiday-planner.git
cd a2a-tripplanner-openai

# Install dependencies
uv sync
```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory with:
```bash
OPENAI_API_KEY=sk-...
SERPER_API_KEY=your-serper-key
```

---

## ▶️ Running the Agents

Each agent is its own microservice.  
Run each in its own terminal tab (or use tmux / VSCode tasks):

```bash
# 1️⃣ Flights Agent
uv run -m flights_crewai.main

# 2️⃣ Hotels Agent
uv run -m hotels_crewai.main

# 3️⃣ Activities Agent
uv run -m activities_crewai.main

# 4️⃣ Budget Agent
uv run -m budget_agent.main

# 5️⃣ Host Agent (Orchestrator)
uv run --active adk web
```
Once all agents are running, open the A2A UI in your browser.
From the interface, select host_agent_latest to start the orchestration process.

When all are running, the Host will automatically resolve the others’ A2A cards via:
```
http://localhost:12021
http://localhost:12022
http://localhost:12023
http://localhost:12024
```

---

## 💬 Sending a Request

You can interact with the Host directly via `curl` or any HTTP client.

```bash
curl -X POST http://localhost:12020/messages:sendMessage   -H "Content-Type: application/json"   -d '{
    "id": "msg-1",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"type": "text", "text": "Plan a 4-day trip from Berlin to Lisbon, budget 2200€"}],
        "messageId": "msg-1",
        "taskId": "task-1",
        "contextId": "ctx-1"
      }
    }
  }'
```

---


## 🧠 Key Learning Points

- How to build a **distributed multi-agent system** using A2A  
- How to mix **LangGraph orchestration** with **CrewAI scraping tools**
- How to ensure **structured JSON responses** between agents  
- How to coordinate **multiple LLM calls** in parallel and merge outputs  
- How to build a **visual UI** around A2A responses  

---

## 📘 License

MIT License © 2025 Hristo Baldzhiyski / A2A Trip Planner Demo

---

## 💡 Credits
- **A2A Protocol** — Agent-to-Agent communication model.
- **CrewAI** — Built-in web scraping/search agents.
- **LangGraph** — Graph-based orchestration for multi-step reasoning.
- **OpenAI** — Core LLM engine for all agents.

---
