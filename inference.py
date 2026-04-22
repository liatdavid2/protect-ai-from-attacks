import os
import time
from typing import Any, Dict, Optional, TypedDict

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

from harmful_content_input_guard.config import EMBEDDING_MODEL_NAME as HARMFUL_EMBEDDING_MODEL_NAME
from harmful_content_input_guard.features import EmbeddingEncoder as HarmfulEmbeddingEncoder
from harmful_content_input_guard.latest_run import get_latest_run_dir as get_latest_harmful_run_dir
from harmful_content_input_guard.model import load_model as load_harmful_model
from prompt_injection.config import EMBEDDING_MODEL_NAME as PROMPT_EMBEDDING_MODEL_NAME
from prompt_injection.features import EmbeddingEncoder as PromptEmbeddingEncoder
from prompt_injection.latest_run import get_latest_run_dir as get_latest_prompt_run_dir
from prompt_injection.model import load_model as load_prompt_model

APP_TITLE = "Secure SLM Gateway"
APP_VERSION = "1.0.0"

DEFAULT_PROMPT_THRESHOLD = float(os.getenv("PROMPT_INJECTION_THRESHOLD", "0.70"))
DEFAULT_HARMFUL_THRESHOLD = float(os.getenv("HARMFUL_CONTENT_THRESHOLD", "0.70"))
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_SMALL_MODEL = os.getenv("DEFAULT_SMALL_MODEL", "qwen2.5:0.5b")
DEFAULT_OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))

app = FastAPI(title=APP_TITLE, version=APP_VERSION)


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model_name: str = Field(default=DEFAULT_SMALL_MODEL)
    prompt_injection_threshold: float = Field(default=DEFAULT_PROMPT_THRESHOLD, ge=0.0, le=1.0)
    harmful_content_threshold: float = Field(default=DEFAULT_HARMFUL_THRESHOLD, ge=0.0, le=1.0)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)


class PromptGuardRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    threshold: float = Field(default=DEFAULT_PROMPT_THRESHOLD, ge=0.0, le=1.0)


class HarmfulGuardRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    threshold: float = Field(default=DEFAULT_HARMFUL_THRESHOLD, ge=0.0, le=1.0)


class GuardResult(BaseModel):
    predicted_label: int
    malicious_probability: float
    threshold: float
    blocked: bool
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    prompt_guard_model_path: str
    harmful_guard_model_path: str
    default_small_model: str
    ollama_base_url: str


class ChatResponse(BaseModel):
    allowed: bool
    prompt_guard: GuardResult
    harmful_guard: GuardResult
    model_name: Optional[str] = None
    response: Optional[str] = None
    block_reason: Optional[str] = None
    total_latency_ms: float
    model_latency_ms: float


class InferenceState(TypedDict, total=False):
    prompt: str
    model_name: str
    prompt_injection_threshold: float
    harmful_content_threshold: float
    temperature: float
    max_tokens: Optional[int]
    prompt_guard: Dict[str, Any]
    harmful_guard: Dict[str, Any]
    allowed: bool
    block_reason: Optional[str]
    response: Optional[str]
    model_latency_ms: float


class SecureGateway:
    def __init__(self) -> None:
        prompt_run_dir = get_latest_prompt_run_dir()
        harmful_run_dir = get_latest_harmful_run_dir()

        self.prompt_guard_model_path = prompt_run_dir / "xgb_prompt_injection.joblib"
        self.harmful_guard_model_path = harmful_run_dir / "xgb_harmful_content.joblib"

        self.prompt_encoder = PromptEmbeddingEncoder(PROMPT_EMBEDDING_MODEL_NAME)
        self.harmful_encoder = HarmfulEmbeddingEncoder(HARMFUL_EMBEDDING_MODEL_NAME)

        self.prompt_guard_model = load_prompt_model(self.prompt_guard_model_path)
        self.harmful_guard_model = load_harmful_model(self.harmful_guard_model_path)

    def run_prompt_guard(self, prompt: str, threshold: float) -> Dict[str, Any]:
        start = time.perf_counter()
        X = self.prompt_encoder.encode([prompt])
        pred = int(self.prompt_guard_model.predict(X)[0])
        proba = float(self.prompt_guard_model.predict_proba(X)[0][1])
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "predicted_label": pred,
            "malicious_probability": round(proba, 6),
            "threshold": threshold,
            "blocked": proba >= threshold,
            "latency_ms": round(latency_ms, 2),
        }

    def run_harmful_guard(self, prompt: str, threshold: float) -> Dict[str, Any]:
        start = time.perf_counter()
        X = self.harmful_encoder.encode([prompt])
        pred = int(self.harmful_guard_model.predict(X)[0])
        proba = float(self.harmful_guard_model.predict_proba(X)[0][1])
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "predicted_label": pred,
            "malicious_probability": round(proba, 6),
            "threshold": threshold,
            "blocked": proba >= threshold,
            "latency_ms": round(latency_ms, 2),
        }

    def prompt_injection_agent(self, state: InferenceState) -> InferenceState:
        return {"prompt_guard": self.run_prompt_guard(state["prompt"], state["prompt_injection_threshold"])}

    def harmful_content_agent(self, state: InferenceState) -> InferenceState:
        return {"harmful_guard": self.run_harmful_guard(state["prompt"], state["harmful_content_threshold"])}

    def decision_agent(self, state: InferenceState) -> InferenceState:
        if state["prompt_guard"]["blocked"]:
            return {"allowed": False, "block_reason": "Prompt blocked by prompt injection guard."}
        if state["harmful_guard"]["blocked"]:
            return {"allowed": False, "block_reason": "Prompt blocked by harmful content guard."}
        return {"allowed": True, "block_reason": None}

    def small_model_agent(self, state: InferenceState) -> InferenceState:
        start = time.perf_counter()
        payload: Dict[str, Any] = {
            "model": state["model_name"],
            "prompt": state["prompt"],
            "stream": False,
            "options": {"temperature": state["temperature"]},
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
            raise HTTPException(status_code=502, detail=f"Failed to call protected model via Ollama: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {"response": data.get("response", ""), "model_latency_ms": round(latency_ms, 2)}

    def route_after_decision(self, state: InferenceState) -> str:
        return "small_model_agent" if state["allowed"] else END

    def build_graph(self):
        graph = StateGraph(InferenceState)
        graph.add_node("prompt_injection_agent", self.prompt_injection_agent)
        graph.add_node("harmful_content_agent", self.harmful_content_agent)
        graph.add_node("decision_agent", self.decision_agent)
        graph.add_node("small_model_agent", self.small_model_agent)
        graph.add_edge(START, "prompt_injection_agent")
        graph.add_edge("prompt_injection_agent", "harmful_content_agent")
        graph.add_edge("harmful_content_agent", "decision_agent")
        graph.add_conditional_edges("decision_agent", self.route_after_decision, {"small_model_agent": "small_model_agent", END: END})
        graph.add_edge("small_model_agent", END)
        return graph.compile()


gateway = SecureGateway()
compiled_graph = gateway.build_graph()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        prompt_guard_model_path=str(gateway.prompt_guard_model_path),
        harmful_guard_model_path=str(gateway.harmful_guard_model_path),
        default_small_model=DEFAULT_SMALL_MODEL,
        ollama_base_url=DEFAULT_OLLAMA_BASE_URL,
    )


@app.post("/prompt-guard", response_model=GuardResult)
def prompt_guard(req: PromptGuardRequest) -> GuardResult:
    return GuardResult(**gateway.run_prompt_guard(req.prompt, req.threshold))


@app.post("/harmful-guard", response_model=GuardResult)
def harmful_guard(req: HarmfulGuardRequest) -> GuardResult:
    return GuardResult(**gateway.run_harmful_guard(req.prompt, req.threshold))


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    total_start = time.perf_counter()
    state: InferenceState = {
        "prompt": req.prompt,
        "model_name": req.model_name,
        "prompt_injection_threshold": req.prompt_injection_threshold,
        "harmful_content_threshold": req.harmful_content_threshold,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "response": None,
        "model_latency_ms": 0.0,
        "block_reason": None,
    }
    final_state = compiled_graph.invoke(state)
    total_latency_ms = (time.perf_counter() - total_start) * 1000.0
    return ChatResponse(
        allowed=final_state["allowed"],
        prompt_guard=GuardResult(**final_state["prompt_guard"]),
        harmful_guard=GuardResult(**final_state["harmful_guard"]),
        model_name=req.model_name if final_state["allowed"] else None,
        response=final_state.get("response"),
        block_reason=final_state.get("block_reason"),
        total_latency_ms=round(total_latency_ms, 2),
        model_latency_ms=final_state.get("model_latency_ms", 0.0),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference:app", host="0.0.0.0", port=8000, reload=True)
