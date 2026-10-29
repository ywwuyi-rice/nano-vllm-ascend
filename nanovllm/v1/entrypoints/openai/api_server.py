import argparse
import asyncio
import os
from contextlib import asynccontextmanager
import sys
from pathlib import Path

# 确保可以导入主版本的 nanovllm 和 v1 的模块
v1_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(v1_path))  # 项目根目录
# v1 目录需要在 Python path 中
if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import fastapi
import uvicorn
from http import HTTPStatus
from fastapi import Request, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response

from nanovllm.config import Config
from nanovllm.v1.engine import AsyncLLMEngine
from nanovllm.v1.entrypoints.openai.protocol import (
    CompletionRequest,
    CompletionResponse,
    ChatCompletionRequest,
    ErrorResponse,
    ModelCard,
    ModelList,
)
from nanovllm.v1.entrypoints.openai.serving_completion import OpenAIServingCompletion
from nanovllm.v1.entrypoints.openai.serving_chat import OpenAIServingChat

from nanovllm.utils.logger import init_logger

logger = init_logger(__name__)

TIMEOUT_KEEP_ALIVE = 5

engine: AsyncLLMEngine = None
serving_completion: OpenAIServingCompletion = None
serving_chat: OpenAIServingChat = None


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    global engine, serving_completion, serving_chat
    logger.info("lifespan starting")

    def parse_args():
        parser = argparse.ArgumentParser(
            description="nano-vllm-ascend OpenAI-Compatible RESTful API server (v1).")
        parser.add_argument("--host", type=str, default="0.0.0.0", help="host name")
        parser.add_argument("--port", type=int, default=8000, help="port number")
        parser.add_argument("--model", type=str, required=True, help="model path")
        parser.add_argument("--served-model-name",
                            type=str,
                            default=None,
                            help="The model name used in the API.")
        parser.add_argument("--max-num-batched-tokens",
                            type=int,
                            default=16384,
                            help="Maximum number of batched tokens")
        parser.add_argument("--max-num-seqs",
                            type=int,
                            default=256,
                            help="Maximum number of sequences")
        parser.add_argument("--max-model-len",
                            type=int,
                            default=4096,
                            help="Maximum model length")
        parser.add_argument("--gpu-memory-utilization",
                            type=float,
                            default=0.9,
                            help="GPU memory utilization")
        parser.add_argument("--tensor-parallel-size",
                            type=int,
                            default=1,
                            help="Tensor parallel size")
        parser.add_argument("--enable-sp",
                            action="store_true",
                            help="Enable sequence parallel (prefill only)")
        parser.add_argument("--hccl-port",
                            type=int,
                            default=28000,
                            help="HCCL port for distributed initialization")
        parser.add_argument("--enforce-eager",
                            action="store_true",
                            default=True,
                            help="Force eager mode execution")
        parser.add_argument("--log-level",
                            type=str,
                            default="info",
                            help="Log level")
        return parser.parse_args()

    args = parse_args()

    hccl_port = getattr(args, 'hccl_port', None) or int(os.getenv('HCCL_PORT', '28000'))

    engine = AsyncLLMEngine(
        model=args.model,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_sp=args.enable_sp,
        hccl_port=hccl_port,
        enforce_eager=args.enforce_eager,
        log_requests=True,
        start_engine_loop=True,
    )

    served_model_name = args.served_model_name or args.model
    serving_completion = OpenAIServingCompletion(
        engine,
        served_model=[served_model_name],
    )
    logger.info(f"serving_completion initialized: {serving_completion}")
    serving_chat = OpenAIServingChat(
        engine,
        served_model=[served_model_name],
    )
    logger.info(f"serving_chat initialized: {serving_chat}")

    logger.info(f"Nano-vLLM-Ascend V1 API server started on {args.host}:{args.port}")
    logger.info(f"Serving model: {served_model_name}")

    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/models")
async def show_available_models():
    global engine
    models = [
        ModelCard(
            id=model_name,
            object="model",
            owned_by="nano-vllm-ascend",
            permission=[],
        )
        for model_name in serving_completion.served_model
    ]
    return ModelList(data=models).model_dump()


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    global serving_completion
    generator = await serving_completion.create_completion(request)

    if isinstance(generator, str):
        return Response(
            content=generator,
            media_type="application/json",
        )
    else:
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
        )


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    global serving_chat
    generator = await serving_chat.create_chat_completion(request, raw_request)

    if isinstance(generator, str):
        return Response(
            content=generator,
            media_type="application/json",
        )
    else:
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
        )


@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        content={
            "object": "error",
            "message": str(exc),
            "type": "internal_error",
            "code": 500,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="nano-vllm-ascend OpenAI-Compatible RESTful API server (v1).")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument("--model", type=str, required=True, help="model path")
    parser.add_argument("--served-model-name",
                        type=str,
                        default=None,
                        help="The model name used in the API.")
    parser.add_argument("--max-num-batched-tokens",
                        type=int,
                        default=16384,
                        help="Maximum number of batched tokens")
    parser.add_argument("--max-num-seqs",
                        type=int,
                        default=256,
                        help="Maximum number of sequences")
    parser.add_argument("--max-model-len",
                        type=int,
                        default=4096,
                        help="Maximum model length")
    parser.add_argument("--gpu-memory-utilization",
                        type=float,
                        default=0.9,
                        help="GPU memory utilization")
    parser.add_argument("--tensor-parallel-size",
                        type=int,
                        default=1,
                        help="Tensor parallel size")
    parser.add_argument("--hccl-port",
                        type=int,
                        default=28000,
                        help="HCCL port for distributed initialization")
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        default=True,
                        help="Force eager mode execution")
    args = parser.parse_args()

    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )
