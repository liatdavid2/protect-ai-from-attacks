# Secure SLM Gateway

A local API gateway that protects a small language model with two input guards before forwarding safe prompts to Ollama:

- Prompt Injection Guard
- Harmful Content Guard

## Project Structure

```text
secure-slm-gateway/
├── artifacts/
├── harmful_content/
├── prompt_injection/
├── inference.py
├── requirements.txt
└── .env.example
```

Each guard stores:

- embeddings cache in `artifacts/<guard_name>/cache/`
- trained runs in `artifacts/<guard_name>/runs/<timestamp>/`

## Python Version

This project is pinned for Python 3.9.

## Setup

```bash
python -m venv .venv
.venv\Scriptsctivate
pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

Fill `HF_TOKEN` in `.env`.

## Train Prompt Injection Guard

```bash
python prompt_injection	rain.py
```

## Train Harmful Content Guard

```bash
python harmful_content	rain.py
```

## Run API

```bash
python inference.py
```

Open Swagger:

```text
http://127.0.0.1:8000/docs
```

## Endpoints

### `GET /health`
Checks that the API is running and shows the loaded artifacts.

### `POST /prompt-guard`
Runs only the prompt injection detector.

### `POST /harmful-guard`
Runs only the harmful content detector.

### `POST /chat`
Flow:

1. prompt injection guard
2. harmful content guard
3. forward safe prompt to Ollama

## Ollama

`/chat` requires local Ollama.

```bash
ollama serve
ollama pull qwen2.5:0.5b
```

## Notes

- `prompt_injection` uses `neuralchemy/Prompt-injection-dataset` with config `core`.
- `harmful_content` uses `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` with `prompt` and `prompt_label` for input moderation.
- If embeddings cache already exists, training reuses it instead of recomputing embeddings.
