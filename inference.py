import os
import time
from typing import Any, Dict, Optional, TypedDict

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

from prompt_injection.config import EMBEDDING_MODEL_NAME
from prompt_injection.features import EmbeddingEncoder
from prompt_injection.latest_run import get_latest_run_dir
from prompt_injection.model import load_model

from dotenv import load_dotenv
load_dotenv()

APP_TITLE = "Secure SLM Gateway"
APP_VERSION = "1.0.0"

DEFAULT_THRESHOLD = float(os.getenv("PROMPT_INJECTION_THRESHOLD", "0.70"))
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_SMALL_MODEL = os.getenv("DEFAULT_SMALL_MODEL", "qwen2.5:0.5b")
DEFAULT_OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))


app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=(
        "API gateway that checks prompts with a prompt injection detector "
        "before forwarding safe requests to a small language model."
    ),
)


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User input prompt")
    model_name: str = Field(
        default=DEFAULT_SMALL_MODEL,
        description="Protected small model name served by Ollama",
    )
    threshold: float = Field(
        default=DEFAULT_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Blocking threshold for the prompt injection detector",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Generation temperature for the protected model",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional maximum number of tokens to generate",
    )


class GuardOnlyRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User input prompt")
    threshold: float = Field(
        default=DEFAULT_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Blocking threshold for the prompt injection detector",
    )


class GuardResult(BaseModel):
    predicted_label: int
    malicious_probability: float
    threshold: float
    blocked: bool
    latency_ms: float


class ChatResponse(BaseModel):
    allowed: bool
    guard: GuardResult
    model_name: Optional[str] = None
    response: Optional[str] = None
    block_reason: Optional[str] = None
    guard_latency_ms: float
    model_latency_ms: float
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    guard_model_path: str
    embedding_model_name: str
    default_small_model: str
    ollama_base_url: str


class InferenceState(TypedDict, total=False):
    prompt: str
    model_name: str
    threshold: float
    temperature: float
    max_tokens: Optional[int]

    guard_predicted_label: int
    guard_malicious_probability: float
    guard_blocked: bool
    guard_latency_ms: float

    allowed: bool
    block_reason: Optional[str]

    response: Optional[str]
    model_latency_ms: float
    total_latency_ms: float


class SecureGateway:
    def __init__(self) -> None:
        latest_run_dir = get_latest_run_dir()
        self.guard_model_path = latest_run_dir / "xgb_prompt_injection.joblib"
        self.encoder = EmbeddingEncoder(EMBEDDING_MODEL_NAME)
        self.guard_model = load_model(self.guard_model_path)

    def prompt_injection_agent(self, state: InferenceState) -> InferenceState:
        start = time.perf_counter()

        X = self.encoder.encode([state["prompt"]])
        pred = int(self.guard_model.predict(X)[0])
        proba = float(self.guard_model.predict_proba(X)[0][1])

        latency_ms = (time.perf_counter() - start) * 1000.0
        blocked = proba >= state["threshold"]

        return {
            "guard_predicted_label": pred,
            "guard_malicious_probability": round(proba, 6),
            "guard_blocked": blocked,
            "guard_latency_ms": round(latency_ms, 2),
        }

    def decision_agent(self, state: InferenceState) -> InferenceState:
        if state["guard_blocked"]:
            return {
                "allowed": False,
                "block_reason": "Prompt blocked by prompt injection guard.",
            }

        return {
            "allowed": True,
            "block_reason": None,
        }

    def small_model_agent(self, state: InferenceState) -> InferenceState:
        start = time.perf_counter()

        payload: Dict[str, Any] = {
            "model": state["model_name"],
            "prompt": state["prompt"],
            "stream": False,
            "options": {
                "temperature": state["temperature"],
            },
        }

        if state.get("max_tokens") is not None:
            payload["options"]["num_predict"] = state["max_tokens"]

        try:
            resp = requests.post(
                f"{DEFAULT_OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=DEFAULT_OLLAMA_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to call protected model via Ollama: {exc}",
            ) from exc

        response_text = data.get("response", "")
        latency_ms = (time.perf_counter() - start) * 1000.0

        return {
            "response": response_text,
            "model_latency_ms": round(latency_ms, 2),
        }

    def route_after_decision(self, state: InferenceState) -> str:
        return "small_model_agent" if state["allowed"] else END

    def build_graph(self):
        graph = StateGraph(InferenceState)

        graph.add_node("prompt_injection_agent", self.prompt_injection_agent)
        graph.add_node("decision_agent", self.decision_agent)
        graph.add_node("small_model_agent", self.small_model_agent)

        graph.add_edge(START, "prompt_injection_agent")
        graph.add_edge("prompt_injection_agent", "decision_agent")
        graph.add_conditional_edges(
            "decision_agent",
            self.route_after_decision,
            {
                "small_model_agent": "small_model_agent",
                END: END,
            },
        )
        graph.add_edge("small_model_agent", END)

        return graph.compile()

    def run_guard_only(self, prompt: str, threshold: float) -> Dict[str, Any]:
        start = time.perf_counter()

        X = self.encoder.encode([prompt])
        pred = int(self.guard_model.predict(X)[0])
        proba = float(self.guard_model.predict_proba(X)[0][1])

        latency_ms = (time.perf_counter() - start) * 1000.0
        blocked = proba >= threshold

        return {
            "predicted_label": pred,
            "malicious_probability": round(proba, 6),
            "threshold": threshold,
            "blocked": blocked,
            "latency_ms": round(latency_ms, 2),
        }


gateway = SecureGateway()
compiled_graph = gateway.build_graph()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        guard_model_path=str(gateway.guard_model_path),
        embedding_model_name=EMBEDDING_MODEL_NAME,
        default_small_model=DEFAULT_SMALL_MODEL,
        ollama_base_url=DEFAULT_OLLAMA_BASE_URL,
    )


@app.post("/guard", response_model=GuardResult)
def guard_only(req: GuardOnlyRequest) -> GuardResult:
    result = gateway.run_guard_only(req.prompt, req.threshold)
    return GuardResult(**result)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    total_start = time.perf_counter()

    initial_state: InferenceState = {
        "prompt": req.prompt,
        "model_name": req.model_name,
        "threshold": req.threshold,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "response": None,
        "model_latency_ms": 0.0,
        "block_reason": None,
    }

    final_state = compiled_graph.invoke(initial_state)
    total_latency_ms = (time.perf_counter() - total_start) * 1000.0

    guard = GuardResult(
        predicted_label=final_state["guard_predicted_label"],
        malicious_probability=final_state["guard_malicious_probability"],
        threshold=req.threshold,
        blocked=final_state["guard_blocked"],
        latency_ms=final_state["guard_latency_ms"],
    )

    return ChatResponse(
        allowed=final_state["allowed"],
        guard=guard,
        model_name=req.model_name if final_state["allowed"] else None,
        response=final_state.get("response"),
        block_reason=final_state.get("block_reason"),
        guard_latency_ms=final_state["guard_latency_ms"],
        model_latency_ms=final_state.get("model_latency_ms", 0.0),
        total_latency_ms=round(total_latency_ms, 2),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("inference:app", host="0.0.0.0", port=8000, reload=True)