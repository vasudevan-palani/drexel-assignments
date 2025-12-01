# Xfinity Triage Assistant (CLI Demo)

A command-line (CLI) application that uses a Large Language Model (LLM) to automatically triage customer issues for Xfinity services.  
This demo showcases natural language understanding, a retrieval-augmented knowledge base, tool-calling for equipment diagnostics, and multi-turn troubleshooting.

## 🚀 Features

### 🗣 Natural Language Chat
Users describe issues in plain English (e.g., “My internet is not working”) and the assistant interprets them without requiring structured inputs.

### 🤖 LLM-Based Triage
The assistant classifies each issue into:
- **Service:** internet, tv, billing, mobile, home_security  
- **Category:** slow_speed, no_connectivity, equipment_fee, etc.  
- **Severity:** low, medium, high, urgent  
- **Recommended Channel:** self_service, chat_agent, technician_visit  
- **Outage Detection:** True/False  

### 📚 RAG (Retrieval-Augmented Generation)
A local file, `triage_knowledge.json`, acts as a simplified knowledge base.  
Relevant entries are retrieved for each query and injected into the LLM prompt.

### 🛠 Tool Calling
The LLM can request:
```
check_equipment_status(account_id)
```
The CLI executes the tool, returns the result to the LLM, and the model updates its triage accordingly.

### 🔄 Multi-turn Conversation
The CLI maintains conversation history, enabling:
- Follow-up questions  
- Updated classifications  
- Tool-triggered responses  
- Escalation transitions (e.g., self_service → technician_visit)

## 📁 Project Structure

```
.
├── main.py                   # Main CLI application
├── triage_knowledge.json     # Triage knowledge base (RAG)
└── README.md                 # Project documentation
```

## 🧩 Requirements

### Python 3.9+
Install dependencies:

```bash
pip install openai
```

Set your OpenAI API key:

**macOS/Linux**
```bash
export OPENAI_API_KEY="your_api_key_here"
```

**Windows**
```bash
set OPENAI_API_KEY=your_api_key_here
```

## ▶️ Running the Application

Start the CLI:

```bash
python main.py
```

Example startup:

```
=== Xfinity Triage Assistant (CLI) ===
Type your issue in natural language.
You can optionally include an account id like: 'Account: 123456'.
Type 'exit' or 'quit' to end.
```

## 🧠 How It Works

### 1️⃣ User Input → Knowledge Retrieval  
`triage_knowledge.json` contains triage scenarios with:
- Symptoms  
- Keywords  
- Steps  
- Default severity  
- Default escalation channel  

The CLI retrieves the top matches and injects them into the system prompt.

### 2️⃣ LLM Triage Decision  
The model returns a structured JSON response:

```json
{
  "triage": {
    "service": "internet",
    "category": "no_connectivity",
    "severity": "high",
    "recommended_channel": "self_service",
    "is_outage_related": false
  },
  "reply": "Troubleshooting steps...",
  "follow_up_questions": ["..."],
  "tool_request": {
    "name": "check_equipment_status",
    "arguments": { "account_id": "1234" }
  }
}
```

### 3️⃣ Tool Execution  
If the model requests the tool, the CLI runs:

```python
check_equipment_status(account_id)
```

Example output:

```json
{
  "gateway_online": true,
  "signal_quality": "good",
  "area_outage_detected": false
}
```

The tool result is fed back into the LLM to refine triage.

### 4️⃣ Escalation Logic  
The assistant may escalate to:
- Chat agent  
- Phone agent  
- Technician visit  

based on symptoms and tool data.

## 💬 Example Interaction

```
You: My internet is not working
Assistant: Please provide your account ID.
You: 3456
[Tool] check_equipment_status called with account_id=3456
Assistant: Equipment is online. Try restarting the modem...
You: I tried restarting, still not working.
Assistant: Scheduling technician visit...
```

## 🛠 Customizing the Knowledge Base

Modify or add entries in `triage_knowledge.json`.

Each entry includes:

```json
{
  "id": "internet_no_connectivity_blinking_orange",
  "service": "internet",
  "keywords": ["no internet", "blinking orange"],
  "triage_steps": ["...", "..."],
  "default_severity": "high",
  "default_channel": "self_service_then_technician"
}
```

The CLI will automatically use your updated knowledge.

## ⚠️ Limitations

- RAG retrieval uses simple keyword matching  
- Tool integration is simulated  
- No real backend connectivity  
- Requires further guardrails for production readiness  

## 🌱 Future Improvements

- Vector-based RAG for semantic retrieval  
- Integration with real backend APIs (equipment, billing, outage systems)  
- Web or mobile interface  
- Improved tool-calling logic and validation  
- Additional knowledge base entries  
- Monitoring & analytics for triage performance  

## 📄 License

This project is for **educational and demonstration purposes** only.
