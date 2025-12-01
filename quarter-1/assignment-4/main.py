#!/usr/bin/env python3
"""
xfinity_triage_cli.py

CLI-based chat application that:
1. Lets a user chat about Xfinity issues in natural language.
2. Uses an LLM to triage issues (service, category, severity, next steps).
3. Uses a local triage knowledge base (RAG-style) loaded from triage_knowledge.json.
4. Exposes a "tool" to the LLM: check_equipment_status(account_id)
   which simulates checking modem/gateway status.

Requirements:
    pip install openai

Environment:
    export OPENAI_API_KEY="YOUR_API_KEY"
"""

import json
import os
import sys
from typing import List, Dict, Any
from openai import OpenAI

# ------------ Config ------------

KNOWLEDGE_FILE = "triage_knowledge.json"
MODEL_NAME = "gpt-4.1-mini"   # or gpt-4o-mini, or any chat model you have


# ------------ Load Knowledge (RAG) ------------

def load_knowledge(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"Knowledge file '{path}' not found. Exiting.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve_relevant_entries(knowledge: List[Dict[str, Any]],
                              query: str,
                              top_k: int = 3) -> List[Dict[str, Any]]:
    """
    VERY simple retrieval: keyword overlap between query and 'keywords' field.
    Enough for demo/assignment.
    """
    query_lower = query.lower()
    scored = []

    for entry in knowledge:
        score = 0
        for kw in entry.get("keywords", []):
            if kw.lower() in query_lower:
                score += 1
        if score > 0:
            scored.append((score, entry))

    # sort by score descending and take top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:top_k]]


def format_knowledge_for_prompt(entries: List[Dict[str, Any]]) -> str:
    """
    Format retrieved knowledge as text for the LLM.
    """
    if not entries:
        return "No specific playbook entries matched. Use your general knowledge."

    blocks = []
    for e in entries:
        block = {
            "id": e.get("id"),
            "service": e.get("service"),
            "symptoms": e.get("symptoms"),
            "triage_steps": e.get("triage_steps"),
            "default_severity": e.get("default_severity"),
            "default_channel": e.get("default_channel")
        }
        blocks.append(block)
    return json.dumps(blocks, indent=2)


# ------------ Tool Implementation ------------

def check_equipment_status(account_id: str) -> Dict[str, Any]:
    """
    Simulated tool that checks customer's equipment status.

    In a real system, this would call an internal API (e.g., Xfinity backend).
    Here we return static/demo data, but it demonstrates the "tool call" pattern.
    """
    # For demo, we pretend this is a live lookup.
    # You can tweak based on account_id if you want variety.
    return {
        "account_id": account_id,
        "gateway_online": True,
        "last_reboot_utc": "2025-11-25T13:45:00Z",
        "signal_quality": "good",
        "area_outage_detected": False,
        "notes": "Device reachable, signal levels within normal range."
    }


# ------------ OpenAI Client & Tools Spec ------------

client = OpenAI()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_equipment_status",
            "description": "Check the customer's modem/gateway status using their account id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {
                        "type": "string",
                        "description": "Xfinity customer account identifier (can be a demo id)."
                    }
                },
                "required": ["account_id"]
            }
        }
    }
]

SYSTEM_PROMPT = """
You are an AI support assistant for Xfinity customers.
You are running inside a CLI tool used by support or by customers.

You have three main goals:
1. Triage each issue (service, category, severity, recommended channel).
2. Use the provided 'knowledge base entries' when relevant.
3. When appropriate, call the function `check_equipment_status` to verify device status.

Your responses MUST follow this JSON schema EXACTLY:

{
  "triage": {
    "service": "internet | tv | mobile | home_security | billing | other",
    "category": "short_snake_case_label_describing_issue",
    "severity": "low | medium | high | urgent",
    "recommended_channel": "self_service | chat_agent | phone_agent | technician_visit",
    "is_outage_related": true_or_false
  },
  "reply": "Short natural language answer for the customer, max 8 sentences.",
  "follow_up_questions": [
    "question 1",
    "question 2"
  ],
  "tool_request": {
    "name": "check_equipment_status | none",
    "arguments": {
      "account_id": "string or empty if not needed"
    }
  }
}

Guidelines:
- Only request the `check_equipment_status` tool if:
  - The issue clearly relates to internet/TV service reliability AND
  - The user has given an account id, or you have just asked for it in a previous turn.
- If you need an account id but do not have one, set tool_request.name = "none"
  and ask the user for their account id (or last 4 digits) in 'reply' / follow_up_questions.
- If not relevant, set tool_request.name = "none".
- Follow up questions are not required when the customers concern is addressed fully.

For 'category', use simple snake_case labels like:
- "no_connectivity", "slow_speed", "equipment_fee", "billing_confusion", "wifi_drops", etc.

You MUST respect and incorporate any 'knowledge_base_entries' text provided to you
in the SAME request. Those are internal playbook rules; treat them as authoritative.

Use only the knowledge from triage_knowledge.json , DO NOT use any other knowledge.
"""

def call_llm(messages: List[Dict[str, str]]) -> str:
    """
    Call the LLM with tools enabled.
    We are NOT using automatic tool-calling here; instead, the model
    will indicate desired tool usage in its JSON via 'tool_request'.
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.3,
    )
    return response.choices[0].message.content


def pretty_print_response(parsed: Dict[str, Any]):
    triage = parsed.get("triage", {})
    reply = parsed.get("reply", "")
    follow_up = parsed.get("follow_up_questions", [])

    print("\n--- Triage Summary --------------------------------")
    print(f"Service:             {triage.get('service', 'unknown')}")
    print(f"Category:            {triage.get('category', 'unknown')}")
    print(f"Severity:            {triage.get('severity', 'unknown')}")
    print(f"Recommended channel: {triage.get('recommended_channel', 'unknown')}")
    print(f"Outage related:      {triage.get('is_outage_related', 'unknown')}")
    print("---------------------------------------------------\n")

    print("Assistant:")
    print(reply)
    if follow_up:
        print("\nFollow-up questions:")
        for q in follow_up:
            print(f"- {q}")
    print("\n")


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Set it via: export OPENAI_API_KEY='your_key_here'")
        return

    knowledge = load_knowledge(KNOWLEDGE_FILE)

    print("=== Xfinity Triage Assistant (CLI) ===")
    print("Type your issue in natural language.")
    print("You can optionally include an account id like: 'Account: 123456'.")
    print("Type 'exit' or 'quit' to end.\n")

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # We track the latest account_id if the user gives one explicitly.
    current_account_id: str | None = None

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not user_input:
            continue

        # naive extraction of "Account: XYZ" pattern
        if "account:" in user_input.lower():
            # Example: "account: 123456"
            parts = user_input.split(":")
            if len(parts) >= 2:
                current_account_id = parts[1].strip()
                print(f"[Info] Captured account_id: {current_account_id}")

        # --- RAG: retrieve knowledge for this turn ---
        kb_entries = retrieve_relevant_entries(knowledge, user_input, top_k=3)
        kb_text = format_knowledge_for_prompt(kb_entries)

        # Add system chunk with knowledge for this turn
        messages.append({
            "role": "system",
            "content": f"Knowledge base entries for this turn:\n{kb_text}"
        })

        # Add user message
        messages.append({"role": "user", "content": user_input})

        # Call LLM
        raw_output = call_llm(messages)

        # Try to parse JSON returned by the model
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            print("\n[Warning] Model returned non-JSON output, showing raw:")
            print(raw_output)
            print()
            messages.append({"role": "assistant", "content": raw_output})
            continue

        # Check for tool request field
        tool_req = parsed.get("tool_request", {}) or {}
        tool_name = tool_req.get("name", "none")
        tool_args = tool_req.get("arguments", {}) or {}

        # If it wants to call check_equipment_status AND we have an account id
        if tool_name == "check_equipment_status":
            account_id = tool_args.get("account_id") or current_account_id
            if not account_id:
                # The model misbehaved: asked for tool without an account id.
                # We just override and ask user for it via plain text.
                print("\n[Info] Tool requested but no account id available.")
                pretty_print_response(parsed)
            else:
                # Actually call the local tool
                tool_result = check_equipment_status(account_id)
                print("\n[Tool] check_equipment_status called with account_id =", account_id)
                print("[Tool] Result:", json.dumps(tool_result, indent=2), "\n")

                # Add tool result into conversation so model can refine answer next turn if needed
                messages.append({
                    "role": "assistant",
                    "content": raw_output  # original JSON response
                })
                messages.append({
                    "role": "system",
                    "content": (
                        "Tool 'check_equipment_status' was executed. "
                        f"Result:\n{json.dumps(tool_result, indent=2)}"
                    )
                })

                # Optionally, ask the model to provide an updated reply using the tool result
                updated_raw = call_llm(messages)
                try:
                    updated_parsed = json.loads(updated_raw)
                    parsed = updated_parsed  # overwrite with refined answer
                except json.JSONDecodeError:
                    # If it fails, we keep the original parsed JSON
                    print("[Warning] Updated response after tool call was not valid JSON.")

                # Append final assistant message for memory
                messages.append({"role": "assistant", "content": json.dumps(parsed)})

                # Display final response
                pretty_print_response(parsed)

                continue  # go to next user input

        # No tool or tool not usable
        messages.append({"role": "assistant", "content": json.dumps(parsed)})
        pretty_print_response(parsed)


if __name__ == "__main__":
    main()
