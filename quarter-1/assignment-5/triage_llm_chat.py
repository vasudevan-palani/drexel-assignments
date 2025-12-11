# triage_llm_chat.py
#
# LLM-only IT Support triage chat with a simple RAG layer.
#
# This uses:
#   - An OpenAI chat model for conversation
#   - A local IT support knowledge base JSON file
#   - Very simple keyword-based retrieval (no embeddings)
#
# Knowledge file format (it_support_knowledge.json):
# [
#   {
#     "id": "KB-001",
#     "area": "Network/VPN",
#     "keywords": ["vpn", "remote", "tunnel", "anyconnect"],
#     "problem_pattern": "User cannot connect to VPN from home",
#     "triage_steps": [
#       "Confirm the user has general internet access.",
#       "Ask them to retry and capture the exact error message."
#     ],
#     "known_fix": "Often resolved by reinstalling the VPN client and clearing cached credentials."
#   },
#   ...
# ]
#
# Commands:
#   /new  - start triaging a new IT support ticket
#   /quit - exit the tool
#
# Prereqs:
#   - pip install openai rich
#   - export OPENAI_API_KEY="YOUR_API_KEY"
#
# Run:
#   python triage_llm_chat.py

import os
import json
from typing import List, Dict, Any

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

# -----------------------------
# CONFIG
# -----------------------------
OPENAI_MODEL = "gpt-4o-mini"
KNOWLEDGE_FILE = "it_support_knowledge.json"  # your IT support KB

client = OpenAI()
console = Console()


# -----------------------------
# Knowledge (RAG) utilities
# -----------------------------
def load_knowledge(path: str) -> List[Dict[str, Any]]:
    """
    Load IT support knowledge entries from JSON file.
    Each entry should at least have: id, keywords, problem_pattern, triage_steps (list), known_fix (string).
    """
    if not os.path.exists(path):
        console.print(
            f"[bold red]Knowledge file '{path}' not found.[/bold red]\n"
            "Create it_support_knowledge.json with IT support entries first."
        )
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve_relevant_entries(
    knowledge: List[Dict[str, Any]],
    query: str,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    VERY simple retrieval: keyword overlap between query and entry['keywords'].
    Enough for a demo / assignment without embeddings.
    """
    query_lower = query.lower()
    scored: List[tuple[int, Dict[str, Any]]] = []

    for entry in knowledge:
        score = 0
        for kw in entry.get("keywords", []):
            if kw and kw.lower() in query_lower:
                score += 1
        if score > 0:
            scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:top_k]]


def format_knowledge_for_prompt(entries: List[Dict[str, Any]]) -> str:
    """
    Format retrieved knowledge entries as JSON-like text for the LLM.
    We keep it structured but not huge.
    """
    if not entries:
        return "No specific knowledge base entries matched. Use general IT support best practices."

    blocks = []
    for e in entries:
        block = {
            "id": e.get("id"),
            "area": e.get("area"),
            "problem_pattern": e.get("problem_pattern"),
            "triage_steps": e.get("triage_steps"),
            "known_fix": e.get("known_fix"),
        }
        blocks.append(block)
    return json.dumps(blocks, indent=2)


# -----------------------------
# LLM call
# -----------------------------
def call_llm(messages: List[Dict[str, str]]) -> str:
    """
    Call OpenAI Chat Completions with the given message history.
    """
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
    )
    return resp.choices[0].message.content


# -----------------------------
# Main triage session
# -----------------------------
def triage_session():
    # Load KB once
    try:
        knowledge = load_knowledge(KNOWLEDGE_FILE)
    except FileNotFoundError:
        return

    console.print(
        Panel(
            "IT Support Triage Assistant (LLM + KB)\n"
            "[bold]/new[/] = new ticket, [bold]/quit[/] = exit.",
            title="IT Support Triage Chat",
        )
    )

    while True:
        incident = Prompt.ask(
            "\n[bold cyan]Enter IT support issue description[/] (or /quit)"
        )
        if incident.strip().lower() in {"quit", "/quit"}:
            console.print("[bold red]Goodbye![/bold red]")
            break

        # First retrieval: based on the initial incident description
        kb_entries = retrieve_relevant_entries(knowledge, incident, top_k=3)
        kb_text = format_knowledge_for_prompt(kb_entries)

        # Show KB hits to the user for transparency
        console.print(
            Panel(
                kb_text,
                title="Knowledge Base Matches",
                style="yellow",
            )
        )

        # Start a new conversation for this IT support ticket
        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are an IT support / helpdesk triage assistant for employees.\n"
                    "You have access to an internal knowledge base of past issues and "
                    "solutions. When knowledge base entries are provided, you should:\n"
                    "- Use them as primary guidance when relevant.\n"
                    "- Adapt the steps to the user's situation.\n"
                    "- Mention when your suggestion is based on a KB article (e.g. 'According to KB-001...').\n\n"
                    "Your job for each ticket:\n"
                    "1. Ask focused clarifying questions about:\n"
                    "   - Device type (laptop/desktop/phone), OS (Windows/macOS/Linux/iOS/Android)\n"
                    "   - Exact error messages or behavior\n"
                    "   - Scope (only this user vs many users)\n"
                    "   - When it started and whether it worked before\n"
                    "   - Network (office/home/VPN, wired vs Wi-Fi)\n"
                    "2. Suggest concrete troubleshooting steps (one step at a time).\n"
                    "3. Identify a likely category from: "
                    "[Access/Accounts, Network/VPN, Hardware, Software/Application, "
                    "Email/Collaboration, Other].\n"
                    "4. Suggest an approximate PRIORITY "
                    "from: [Low, Medium, High, Urgent].\n"
                    "5. When enough information is available, summarize:\n"
                    "   - Category\n"
                    "   - Suspected root cause or area\n"
                    "   - Suggested priority\n"
                    "   - Next steps and who should own it (e.g., Service Desk L1, App Team, Network Team).\n\n"
                    "Always keep explanations clear and non-technical when possible."
                ),
            },
            {
                "role": "system",
                "content": (
                    "Knowledge base entries for this ticket:\n"
                    f"{kb_text}"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Here is a new IT support issue reported by a user:\n"
                    f"{incident}\n\n"
                    "Using the knowledge base entries above when relevant, "
                    "please start triaging this IT ticket. Ask me clarifying questions "
                    "and suggest next steps. Also start forming a view on category and "
                    "priority as we gather more details."
                ),
            },
        ]

        first_reply = call_llm(messages)
        messages.append({"role": "assistant", "content": first_reply})
        console.print(Panel(first_reply, title="Assistant", style="green"))

        # Multi-turn IT support triage loop
        while True:
            user_msg = Prompt.ask(
                "[bold magenta]You[/] (details/answers, /new, or /quit)"
            ).strip()

            if user_msg.lower() in {"/quit", "quit"}:
                console.print("[bold red]Goodbye![/bold red]")
                return
            if user_msg.lower() in {"/new", "new"}:
                # Break to outer loop to start a new ticket
                break

            # RAG again each turn: use the original incident + new details
            combined_query = f"{incident}\n\n{user_msg}"
            kb_entries = retrieve_relevant_entries(knowledge, combined_query, top_k=3)
            kb_text = format_knowledge_for_prompt(kb_entries)

            console.print(
                Panel(
                    kb_text,
                    title="Updated Knowledge Base Matches",
                    style="yellow",
                )
            )

            # Inject updated KB as a system message (so LLM sees latest matches)
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Updated knowledge base entries based on the latest details:\n"
                        f"{kb_text}"
                    ),
                }
            )

            messages.append({"role": "user", "content": user_msg})
            reply = call_llm(messages)
            messages.append({"role": "assistant", "content": reply})
            console.print(Panel(reply, title="Assistant", style="green"))


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        console.print(
            "[bold red]ERROR:[/] OPENAI_API_KEY env var not set. "
            "Run: export OPENAI_API_KEY='your_key_here'"
        )
    else:
        triage_session()
