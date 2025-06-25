# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
import asyncio
import uuid
import os
import shutil
from video_llm import video_llm
from fastapi.middleware.cors import CORSMiddleware
import traceback
import yaml

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Load configuration
config = load_config()
DATA_DIR = config['egolife']['videollm']['data_dir']
LLAVA_PORT = config['egolife']['videollm']['llava_port']
app = FastAPI(
    title="Video LLM API",
    description="API for analyzing video content using language models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class VideoLLMRequest(BaseModel):
    question: str
    range: str
    identity: str

class VideoLLMResponse(BaseModel):
    query_time: str
    question: str
    answer: str
    exact_match: bool
    length: float

class ErrorResponse(BaseModel):
    error: str

def process_single_request(item: VideoLLMRequest) -> VideoLLMResponse:
    temp_cache_dir = f"/tmp/cache/task_{uuid.uuid4()}"
    os.makedirs(temp_cache_dir, exist_ok=True)
    try:
        response = video_llm(
            question=item.question,
            range=item.range,
            identity=item.identity,
            cache_dir=temp_cache_dir,
            data_dir=DATA_DIR
        )
        return VideoLLMResponse(**response)
    finally:
        shutil.rmtree(temp_cache_dir)

@app.post(
    "/video_llm",
    response_model=Union[VideoLLMResponse, str, List[Union[VideoLLMResponse, str, None]]]
)
async def video_llm_api(request: Union[VideoLLMRequest, List[Union[VideoLLMRequest, None, str]]]):
    try:
        if isinstance(request, list):
            results = []
            for item in request:
                if item is None or item == "" or (isinstance(item, str) and item.strip() == ""):
                    results.append(None)
                else:
                    if isinstance(item, dict):
                        item = VideoLLMRequest(**item)
                    if not isinstance(item, VideoLLMRequest):
                        results.append(f"error: invalid item type {type(item)}")
                        continue
                    try:
                        # Must wait sequentially, cannot be concurrent
                        result = await asyncio.to_thread(process_single_request, item)
                        results.append(result)
                    except Exception as e:
                        print("Exception occurred:", str(e))
                        traceback.print_exc()
                        results.append(str(e))
            return results
        else:
            if isinstance(request, dict):
                request = VideoLLMRequest(**request)
            try:
                result = await asyncio.to_thread(process_single_request, request)
                return result
            except Exception as e:
                print("Exception occurred:", str(e))
                traceback.print_exc()
                return str(e)
    except Exception as e:
        print("Exception occurred:", str(e))
        traceback.print_exc()
        return str(e)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=LLAVA_PORT)
