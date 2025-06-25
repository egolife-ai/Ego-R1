# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
import asyncio
import uuid
import os
import shutil
from video_llm import video_llm_with_client, get_gemini_client_with_key, clear_gemini
from fastapi.middleware.cors import CORSMiddleware
import traceback
from google.genai.errors import ServerError
from concurrent.futures import ThreadPoolExecutor
import functools
import argparse
import yaml
call_counter = 0
CLEAR_EVERY_N = 100

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Load configuration
config = load_config()
DATA_DIR = config['egoschema']['videollm']['data_dir']
GEMINI_API_KEYS = config['egoschema']['videollm']['gemini_api_keys']
GEMINI_PORT = config['egoschema']['videollm']['gemini_port']

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
    vid_id: str  

class VideoLLMResponse(BaseModel):
    query_time: str
    question: str
    answer: str
    exact_match: bool
    length: float

class ErrorResponse(BaseModel):
    error: str




async def process_single_request(item: VideoLLMRequest, api_key: str,data_dir: str) -> VideoLLMResponse:
    import random
    import random
    
    temp_cache_dir = f"./tmp/cache/task_{uuid.uuid4()}"
    os.makedirs(temp_cache_dir, exist_ok=True)
    

    tried_keys = set()
    last_exception = None
    try:
        for attempt in range(3):
            try:
                tried_keys.add(api_key)
                print(f"Using API key: {api_key}")
              


                response = video_llm_with_client(
                    question=item.question,
                    range_=item.range,
                    cache_dir=temp_cache_dir,
                    gemini_api_key=api_key,
                    data_dir=data_dir
                )
                return VideoLLMResponse(**response)
            except ServerError as e:
                
                if hasattr(e, 'code') and e.code == 503:
                    print(f"ServerError 503 encountered with key {api_key}, attempt {attempt+1}/3. Retrying with new key...")
                    last_exception = e
                   
                    available_keys = [k for k in GEMINI_API_KEYS if k not in tried_keys]
                    if not available_keys:
                        available_keys = GEMINI_API_KEYS  
                    api_key = random.choice(available_keys)
                    continue
                else:
                    raise
            except Exception as e:
                last_exception = e
                break
        
        if last_exception:
            print(f"All retries failed. Last exception: {last_exception}")
            raise last_exception
        raise RuntimeError("process_single_request failed unexpectedly without exception.")
    except Exception as e:
        print(f"Exception occurred: {e}")
        traceback.print_exc()
        return str(e)
    finally:
        shutil.rmtree(temp_cache_dir)
        print(f"Cleared cache directory: {temp_cache_dir}")

@app.post(
    "/video_llm",
    response_model=Union[VideoLLMResponse, str, List[Union[VideoLLMResponse, str, None]]]
)
async def video_llm_api(request: Union[VideoLLMRequest, List[Union[VideoLLMRequest, None, str]]]):
    import random
    global call_counter
    import random

    call_counter += 1
    print(f"[INFO] /video_llm API call count: {call_counter}")
    
    if call_counter % CLEAR_EVERY_N == 0:
        print(f"[INFO] Triggering clear_gemini after {CLEAR_EVERY_N} API calls.")
       
        temp_key = random.choice(GEMINI_API_KEYS)
        video_client = get_gemini_client_with_key(temp_key)
        clear_gemini(video_client)
    try:
        if isinstance(request, list):
            tasks = []
            index_map = []
            valid_items = []
            for idx, item in enumerate(request):
                if item is None or item == "" or (isinstance(item, str) and item.strip() == ""):
                    tasks.append(None)
                else:
                    if isinstance(item, dict):
                        item = VideoLLMRequest(**item)
                    valid_items.append(item)
                    tasks.append("PENDING")  
                    index_map.append(idx)
            num_tasks = len(valid_items)
            num_keys = len(GEMINI_API_KEYS)
            api_keys = []
            if num_tasks <= num_keys:
                api_keys = random.sample(GEMINI_API_KEYS, num_tasks)
                print(f"Using {num_tasks} API keys: {api_keys}")
            else:
                api_keys = [random.choice(GEMINI_API_KEYS) for _ in range(num_tasks)]
            real_tasks = []
            key_idx = 0
            for i, task in enumerate(tasks):
                if task is None:
                    continue
                item = valid_items[key_idx]
                api_key = api_keys[key_idx]
                
                data_dir = f"{DATA_DIR}/{item.vid_id}"
                real_tasks.append(functools.partial(asyncio.run, process_single_request(item, api_key,data_dir)))
                key_idx += 1
            results = []
            if real_tasks:
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=len(real_tasks)) as executor:
                    real_results = await asyncio.gather(
                        *[loop.run_in_executor(executor, task) for task in real_tasks],
                        return_exceptions=True
                    )
            else:
                real_results = []
            result_idx = 0
            for task in tasks:
                if task is None:
                    results.append(None)
                else:
                    result = real_results[result_idx]
                    print(result)
                    if isinstance(result, Exception):
                        results.append(str(f"error: {result}"))
                    else:
                        results.append(result)
                    result_idx += 1
            print(results)
            return results
        else:
            print(request)
            if isinstance(request, dict):
                request = VideoLLMRequest(**request)
            import random
            api_key = random.choice(GEMINI_API_KEYS)
            try:
                data_dir = f"{DATA_DIR}/{request.vid_id}"
                result = await process_single_request(request, api_key,data_dir)
                return result
            except Exception as e:
                print("Exception occurred:", str(e))
                traceback.print_exc()
                return str(e)
    except ValueError as e:
        print("Exception occurred:", str(e))
        traceback.print_exc()
        return str(e)
    except Exception as e:
        print("Exception occurred:", str(e))
        traceback.print_exc()
        return str(e)

def start_server( port: int = 6060):
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start RAG API for a specific video_id")
   
    start_server(port=GEMINI_PORT)
