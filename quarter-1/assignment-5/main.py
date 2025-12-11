"""
incident_it_triage_cli.py

Pure CLI IT Support ticket triage tool with:
- CNN-based PRIORITY prediction (low / medium / high / urgent, etc.)
- LLM-based triage chat
- Simple RAG using a local IT Support knowledge base JSON file
- Live summaries after each turn
- Simple commands: /new, /quit

Components:
  - CNN model:
      priority_cnn_model.keras, priority_tokenizer.pkl, priority_label_encoder.pkl
  - Knowledge base:
      it_support_knowledge.json (simple keyword-based RAG)

Prereqs:
  - Run train_cnn_priority.py first to create:
      priority_cnn_model.keras, priority_tokenizer.pkl, priority_label_encoder.pkl
  - Create it_support_knowledge.json with IT support entries.
  - Set OPENAI_API_KEY env var.

Run:
  python incident_it_triage_cli.py
"""

import os
import json
import pickle
import logging
from typing import List, Dict, Optional, Any, Tuple

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from openai import OpenAI

# -----------------------------
# LOGGING
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    filename="incident_cli_app.log",
    filemode="a",
)
logger = logging.getLogger("incident_it_triage_cli")

# -----------------------------
# CONFIG
# -----------------------------
PRIORITY_MODEL_PATH = "priority_cnn_model.keras"
TOKENIZER_PATH = "priority_tokenizer.pkl"
LABEL_ENCODER_PATH = "priority_label_encoder.pkl"

# MUST match the value used in train_cnn_priority.py
MAX_SEQ_LEN = 40

OPENAI_MODEL = "gpt-4o-mini"

# RAG knowledge file
KNOWLEDGE_FILE = "it_support_knowledge.json"

client = OpenAI()


# =============================
# CNN priority model helpers
# =============================
def load_priority_model():
    logger.info("Loading priority model and artifacts.")
    if not (
        os.path.exists(PRIORITY_MODEL_PATH)
        and os.path.exists(TOKENIZER_PATH)
        and os.path.exists(LABEL_ENCODER_PATH)
    ):
        logger.error(
            "Model or artifacts missing: %s, %s, %s",
            PRIORITY_MODEL_PATH,
            TOKENIZER_PATH,
            LABEL_ENCODER_PATH,
        )
        raise FileNotFoundError(
            "Priority model or artifacts not found.\n"
            "Run train_cnn_priority.py first."
        )

    try:
        model = load_model(PRIORITY_MODEL_PATH)
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
        with open(LABEL_ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)
        logger.info("Priority model and artifacts loaded successfully.")
    except Exception:
        logger.exception("Failed to load priority model or artifacts.")
        raise

    return model, tokenizer, label_encoder


def texts_to_padded_sequences(tokenizer, texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(
        sequences, maxlen=MAX_SEQ_LEN, padding="post", truncating="post"
    )
    return padded


def predict_priority(model, tokenizer, label_encoder, description: str) -> str:
    logger.info(
        "Predicting priority for description: %s",
        description[:200].replace("\n", " "),
    )
    X = texts_to_padded_sequences(tokenizer, [description])
    probs = model.predict(X, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = label_encoder.inverse_transform([idx])[0]
    logger.info("Predicted priority: %s (probs=%s)", label, probs)
    return label


# =============================
# RAG / Knowledge utilities
# =============================
def load_knowledge(path: str) -> List[Dict[str, Any]]:
    """
    Load IT support knowledge entries from JSON file.
    Each entry should at least have: id, keywords, problem_pattern, triage_steps (list), known_fix (string).
    """
    if not os.path.exists(path):
        logger.error("Knowledge file '%s' not found.", path)
        raise FileNotFoundError(
            f"Knowledge file '{path}' not found. "
            "Create it_support_knowledge.json with IT support entries first."
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info("Loaded %d knowledge entries from %s", len(data), path)
    return data


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
    scored: List[Tuple[int, Dict[str, Any]]] = []

    for entry in knowledge:
        score = 0
        for kw in entry.get("keywords", []):
            if kw and kw.lower() in query_lower:
                score += 1
        if score > 0:
            scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = [e for _, e in scored[:top_k]]

    logger.info(
        "RAG retrieved %d entries (top_k=%d) for query snippet='%s'",
        len(results),
        top_k,
        query[:200].replace("\n", " "),
    )
    return results


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


# =============================
# LLM helpers
# =============================
def call_llm(messages: List[Dict[str, str]]) -> str:
    logger.info("Calling LLM model=%s with %d messages.", OPENAI_MODEL, len(messages))
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
        )
        content = resp.choices[0].message.content
        logger.info("LLM call succeeded. Response length=%d chars.", len(content or ""))
        return content
    except Exception:
        logger.exception("Error in call_llm.")
        raise


def summarize_incident(
    incident_id: str,
    description: str,
    priority: str,
    chat_history: List[Dict[str, str]],
) -> str:
    logger.info(
        "Summarizing ticket %s with %d chat messages.",
        incident_id,
        len(chat_history),
    )
    # Use only last few turns to keep prompt small
    recent = chat_history[-6:]
    convo_text = "\n\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in recent
    )

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are summarizing an ongoing IT support/helpdesk ticket triage conversation.\n"
                "Write a concise summary (3–6 bullet points) that covers:\n"
                "- What the IT issue is (device/app/account/network/etc.)\n"
                "- Who is impacted (single user, team, whole site, remote users, etc.)\n"
                "- Key troubleshooting steps tried and their outcomes\n"
                "- Current working hypothesis or likely root cause category\n"
                "- Current status (e.g., open, in progress, workaround in place, resolved)\n"
                "- Next concrete actions (what the IT agent should do next or who to escalate to)\n"
                "Keep it factual, short, and focused on IT support context."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Ticket ID: {incident_id}\n"
                f"User description: {description}\n"
                f"Predicted priority: {priority}\n\n"
                f"Conversation so far:\n{convo_text}"
            ),
        },
    ]
    try:
        summary = call_llm(messages)
        logger.info(
            "Summary for %s generated. Length=%d chars.",
            incident_id,
            len(summary or ""),
        )
        return summary
    except Exception as e:
        logger.exception("Error generating summary for %s.", incident_id)
        return f"(Error generating summary: {e})"


def initial_triage_messages(
    description: str,
    priority: str,
    kb_text: str,
) -> List[Dict[str, str]]:
    logger.info(
        "Building initial IT triage messages. Priority=%s, desc(start)=%s",
        priority,
        description[:200].replace("\n", " "),
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an IT support / helpdesk assistant for an internal IT service desk.\n"
                "You handle tickets such as:\n"
                "- Password/account access issues\n"
                "- Laptop/desktop problems (performance, login, OS issues)\n"
                "- Corporate applications (email, collaboration tools, CRM, ERP, etc.)\n"
                "- VPN / Wi-Fi / network connectivity\n"
                "- Printers and peripherals\n\n"
                "You have access to an internal knowledge base of past issues and solutions. "
                "When knowledge base entries are provided, you should:\n"
                "- Use them as primary guidance when relevant.\n"
                "- Adapt the steps to the user's situation.\n"
                "- Mention when your suggestion is based on a KB article (e.g. 'According to KB-001...').\n\n"
                "You are given the user's ticket description and an ML-predicted PRIORITY "
                "(low / medium / high / urgent). Use this as a starting point, "
                "but correct it if it seems wrong.\n\n"
                "For each response you give:\n"
                "- Ask focused clarifying questions where needed\n"
                "- Suggest concrete troubleshooting steps the IT agent can try\n"
                "- Propose a probable category (e.g., Account, Device, Network, App, Hardware)\n"
                "- Indicate if the priority seems appropriate or should be raised/lowered\n"
                "- Suggest whether to resolve at L1, or escalate to L2/L3 or a specific team\n"
                "Be practical and concise. Avoid long essays."
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
                f"User's ticket description:\n{description}\n\n"
                f"Predicted priority: {priority}\n\n"
                "Using the knowledge base entries above when relevant, "
                "please start triaging this IT support ticket. Ask me clarifying questions "
                "and suggest next steps. Also start forming a view on category and "
                "priority as we gather more details."
            ),
        },
    ]


# =============================
# CLI helpers
# =============================
def print_banner():
    print("=" * 70)
    print(" IT Support Ticket Triage CLI (CNN + LLM + RAG)")
    print("=" * 70)
    print("Commands:")
    print("  /new   - start a new ticket")
    print("  /quit  - exit the tool")
    print("- First, type an end-user ticket description to start.")
    print("- Then chat with the assistant; a summary is updated each turn.\n")


def print_incident_header(
    incident_id: str,
    description: str,
    priority: str,
):
    print("\n" + "-" * 70)
    print(f"TICKET: {incident_id}")
    print("-" * 70)
    print(f"Priority (ML-predicted): {priority}")
    print("User description:")
    print(description)
    print("-" * 70)


def print_assistant(message: str):
    print("\n[Assistant]")
    print(message)


def print_summary(summary: str):
    print("\n[Ticket Summary]")
    print(summary)
    print("-" * 70)


# =============================
# Main CLI loop
# =============================
def run_cli():
    # Check env
    if not os.getenv("OPENAI_API_KEY"):
        msg = (
            "ERROR: OPENAI_API_KEY env var not set.\n"
            "Run: export OPENAI_API_KEY='your_key_here'"
        )
        logger.error("OPENAI_API_KEY not set.")
        print(msg)
        return

    # Load model
    try:
        model, tokenizer, label_encoder = load_priority_model()
    except Exception as e:
        print(f"Error loading priority model: {e}")
        return

    # Load knowledge base
    try:
        knowledge = load_knowledge(KNOWLEDGE_FILE)
    except FileNotFoundError as e:
        print(f"Error loading knowledge base: {e}")
        return

    incident_counter = 1
    current_incident_id: Optional[str] = None
    current_description: Optional[str] = None
    current_priority: Optional[str] = None
    chat_history: List[Dict[str, str]] = []
    current_summary: str = ""

    print_banner()

    while True:
        try:
            if current_incident_id is None:
                user_input = input("Enter IT ticket description (or /quit): ").strip()
            else:
                user_input = input("You (/new, /quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            logger.info("CLI interrupted. Exiting.")
            break

        # Commands
        if user_input.lower() in {"/quit", "quit"}:
            logger.info("Quit command received. Exiting.")
            print("Goodbye.")
            break

        if user_input.lower() in {"/new", "new"}:
            logger.info("New ticket command received. Resetting state.")
            current_incident_id = None
            current_description = None
            current_priority = None
            current_summary = ""
            chat_history = []
            print("\n--- New IT ticket started. Enter a description. ---")
            continue

        if not user_input:
            continue

        # -------------------------
        # New TICKET
        # -------------------------
        if current_incident_id is None:
            # First turn: create ticket, predict priority, initial LLM triage
            current_incident_id = f"TIC-{incident_counter:04d}"
            incident_counter += 1
            current_description = user_input

            logger.info(
                "New IT ticket created: %s, description(start)=%s",
                current_incident_id,
                current_description[:200].replace("\n", " "),
            )

            print("\n[Status] Predicting priority (CNN)...")
            try:
                current_priority = predict_priority(
                    model, tokenizer, label_encoder, current_description
                )
            except Exception as e:
                logger.exception(
                    "Error predicting priority for %s.",
                    current_incident_id,
                )
                current_priority = "UNKNOWN"
                print(f"[Status] Error predicting priority: {e}")

            print_incident_header(
                current_incident_id,
                current_description,
                current_priority,
            )

            # Initial RAG retrieval
            kb_entries = retrieve_relevant_entries(
                knowledge, current_description, top_k=3
            )
            kb_text = format_knowledge_for_prompt(kb_entries)

            print("[Status] Knowledge base matches:")
            print(kb_text)
            print("-" * 70)

            print("[Status] Getting initial IT triage from LLM...")
            messages = initial_triage_messages(
                current_description,
                current_priority,
                kb_text,
            )
            try:
                first_reply = call_llm(messages)
            except Exception as e:
                logger.exception(
                    "Error calling LLM for initial triage of %s.",
                    current_incident_id,
                )
                first_reply = f"(Error calling LLM: {e})"

            # For chat_history we only store user/assistant roles,
            # system messages are rebuilt each turn.
            chat_history = [
                {"role": "user", "content": current_description},
                {"role": "assistant", "content": first_reply},
            ]
            logger.info(
                "Initial triage reply for %s: %s",
                current_incident_id,
                first_reply[:300].replace("\n", " "),
            )

            print_assistant(first_reply)

            print("[Status] Updating ticket summary...")
            current_summary = summarize_incident(
                current_incident_id,
                current_description,
                current_priority,
                chat_history,
            )
            logger.info(
                "Summary after initial triage for %s: %s",
                current_incident_id,
                (current_summary or "")[:300].replace("\n", " "),
            )

            print_summary(current_summary)
            print("[Status] Ready – triaging", current_incident_id)
            continue

        # -------------------------
        # ONGOING CHAT
        # -------------------------
        logger.info(
            "User message for %s: %s",
            current_incident_id,
            user_input[:300].replace("\n", " "),
        )
        chat_history.append({"role": "user", "content": user_input})

        # RAG again each turn: use original description + latest user input
        combined_query = f"{current_description}\n\n{user_input}"
        kb_entries = retrieve_relevant_entries(
            knowledge,
            combined_query,
            top_k=3,
        )
        kb_text = format_knowledge_for_prompt(kb_entries)

        print("[Status] Updated knowledge base matches:")
        print(kb_text)
        print("-" * 70)

        # Rebuild context for LLM call (full chat)
        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are an IT support / helpdesk assistant helping an IT agent triage an active ticket.\n"
                    "The conversation below is between the IT agent (user) and you. "
                    "At each turn you should:\n"
                    "- Interpret the user's latest message in context of the ticket\n"
                    "- Ask any necessary clarifying questions\n"
                    "- Suggest concrete troubleshooting steps and checks\n"
                    "- Indicate likely category (Account, Device, Network, App, Hardware, Other)\n"
                    "- Suggest whether to resolve at L1 or escalate (and to which team level)\n"
                    "Keep responses concise and practical. Avoid unnecessary theory."
                ),
            },
            {
                "role": "system",
                "content": (
                    "Updated knowledge base entries based on the latest details:\n"
                    f"{kb_text}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Ticket ID: {current_incident_id}\n"
                    f"User description: {current_description}\n"
                    f"Predicted priority: {current_priority}\n\n"
                    "We will now continue the IT ticket triage conversation."
                ),
            },
        ]
        for msg in chat_history:
            messages.append(msg)

        print("[Status] Getting LLM reply...")
        try:
            reply = call_llm(messages)
        except Exception as e:
            logger.exception(
                "Error calling LLM for chat turn on %s.",
                current_incident_id,
            )
            reply = f"(Error calling LLM: {e})"

        logger.info(
            "Assistant reply for %s: %s",
            current_incident_id,
            reply[:300].replace("\n", " "),
        )
        chat_history.append({"role": "assistant", "content": reply})

        print_assistant(reply)

        print("[Status] Updating ticket summary...")
        current_summary = summarize_incident(
            current_incident_id,
            current_description,
            current_priority,
            chat_history,
        )
        logger.info(
            "Updated summary for %s: %s",
            current_incident_id,
            (current_summary or "")[:300].replace("\n", " "),
        )

        print_summary(current_summary)
        print("[Status] Ready – triaging", current_incident_id)


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    logger.info("Starting IT Support Ticket Triage CLI (CNN + LLM + RAG).")
    run_cli()
    logger.info("IT Support Ticket Triage CLI exited.")
