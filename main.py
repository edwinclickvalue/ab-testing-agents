import asyncio
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent import run_agent_test

load_dotenv()

app = FastAPI(title="AB Agent Tester")


class TestRequest(BaseModel):
    url: str
    goal: str = "add product to cart"
    persona: str = "casual shopper"


class ABTestRequest(BaseModel):
    url_a: str
    url_b: str
    goal: str = "add product to cart"
    persona: str = "casual shopper"


@app.post("/test")
async def test_single(req: TestRequest):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
    result = await run_agent_test(req.url, req.goal, req.persona)
    return result


@app.post("/ab-test")
async def ab_test(req: ABTestRequest):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

    result_a, result_b = await asyncio.gather(
        run_agent_test(req.url_a, req.goal, req.persona),
        run_agent_test(req.url_b, req.goal, req.persona),
    )

    winner = None
    if result_a.success and not result_b.success:
        winner = "A"
    elif result_b.success and not result_a.success:
        winner = "B"
    elif result_a.success and result_b.success:
        winner = "A" if result_a.steps <= result_b.steps else "B"

    return {
        "winner": winner,
        "version_a": result_a,
        "version_b": result_b,
    }


@app.get("/health")
def health():
    return {"status": "ok"}
