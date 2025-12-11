# IT Support Ticket Triage (CNN + LLM + RAG)

This folder contains a simple CLI tool for triaging IT support tickets using:

- A CNN model to predict ticket priority (`train_cnn_priority.py` → `priority_cnn_model.keras`, `priority_tokenizer.pkl`, `priority_label_encoder.pkl`).
- A small, local knowledge base (`it_support_knowledge.json`) used for simple RAG (keyword overlap retrieval).
- An LLM-based triage assistant that calls OpenAI-style chat completions (wrapper used in `main3.py`).

Files
- `main3.py` - CLI triage application (entry point). Run this after you have the trained model artifacts and an `OPENAI_API_KEY`.
- `train_cnn_priority.py` - Script to train a CNN priority classifier from a CSV dataset.
- `it_support_knowledge.json` - Local knowledge base used by the RAG component.

Requirements / Prerequisites

- Python 3.8+ (tested in a modern Python 3.10+ environment).
- A working venv is recommended.
- The project depends on common ML libraries (TensorFlow/Keras, scikit-learn, pandas, numpy) and an OpenAI client.

A `requirements.txt` file is expected in this folder. If you already generated one, install dependencies from it:

```bash
pip install -r requirements.txt
```

If you don't yet have a `requirements.txt`, here is a minimal example you can use (adjust versions for your platform):

```
tensorflow>=2.10
scikit-learn
pandas
numpy
openai

# Optional utilities
python-dotenv
```

Setup

1. Create and activate a venv (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# install the packages you need, e.g.:
pip install tensorflow scikit-learn pandas numpy openai
```

2. Ensure `it_support_knowledge.json` is present in this folder (it is included here).

3. Export your OpenAI API key into the environment (this tool uses an OpenAI-compatible client and expects the `OPENAI_API_KEY` env var):

```bash
export OPENAI_API_KEY="sk-..."
```

Training the CNN priority model (optional)

If you do not already have the model artifacts (`priority_cnn_model.keras`, `priority_tokenizer.pkl`, `priority_label_encoder.pkl`), train the model with the included dataset `incidents.csv` (downloaded from Kaggle):

1. The repository includes `incidents.csv` (Kaggle IT Support Ticket data). The default `train_cnn_priority.py` uses `DATA_PATH = "it_support_ticket_data.csv"` — update it to `incidents.csv` or set `DATA_PATH = "incidents.csv"` inside `train_cnn_priority.py`.

2. Ensure `TEXT_COL` and `LABEL_COL` in `train_cnn_priority.py` match the column names in `incidents.csv` (common names are `Body`/`Description` and `Priority`).

3. Run the training script:

```bash
python train_cnn_priority.py
```

4. After training completes you should see the saved files:
- `priority_cnn_model.keras`
- `priority_tokenizer.pkl`
- `priority_label_encoder.pkl`

These files are used by the CLI to predict ticket priority.

Running the CLI triage tool (`main3.py`)

1. Ensure the model artifact files and `it_support_knowledge.json` are present in the same folder as `main3.py`.
2. Ensure `OPENAI_API_KEY` is set in your environment.
3. Run the CLI:

```bash
python main3.py
```

What to expect

- On startup the CLI checks for `OPENAI_API_KEY` and the priority model artifacts.
- Enter a short or long ticket description when prompted. The CLI will:
  - Predict a priority using the CNN model.
  - Run a simple RAG (keyword overlap) against `it_support_knowledge.json` to find relevant KB entries.
  - Call the LLM to generate an initial triage response and will show a short ticket summary.
- You can then chat with the assistant. Commands:
  - `/new` - start a new ticket (clears state)
  - `/quit` - exit the CLI

Quick test (without training the model)

If you don't want to train a CNN but still want to test the LLM and RAG parts, you can do one of the following:

- Create dummy artifacts that match the expected API shape (advanced), or
- Temporarily modify `main3.py` to skip loading the CNN and set `current_priority` to a default (e.g., `"medium"`).

Example quick test:

1. Edit `main3.py` and inside `run_cli()` before calling `load_priority_model()` add the following lines to skip the model load (for quick LLM/RAG testing only):

```python
# Quick dev/testing: skip CNN priority prediction
# model, tokenizer, label_encoder = None, None, None
# current_priority = "medium"
```

2. Run `python main3.py` and enter a sample ticket description such as:

```
User cannot connect to VPN from home. AnyConnect shows "connection failed".
```

3. The CLI should show KB matches from `it_support_knowledge.json` and call the LLM to provide troubleshooting steps and a summary.

Troubleshooting

- Error: `OPENAI_API_KEY not set` — export your API key as shown above.
- Error loading model artifacts — confirm the files `priority_cnn_model.keras`, `priority_tokenizer.pkl`, and `priority_label_encoder.pkl` exist in this folder (or train using `train_cnn_priority.py`).
- LLM API errors — verify network connectivity, API key limits, and that your OpenAI client package version is compatible with the code.

Security and cost notes

- The CLI will send user ticket text and some KB context to the LLM service; avoid sending PII or sensitive content in public or shared environments.
- LLM calls may incur usage costs. Use a small model for development and monitor your billing.

Extending this project

- Replace the keyword RAG with an embedding-based retriever for higher-quality matches.
- Add a small web UI or Slack bot wrapper around `main3.py`.
- Add unit tests for the tokenizer/prediction pipeline and RAG functions.

License

This repository is for educational/demo use. No license is specified.
