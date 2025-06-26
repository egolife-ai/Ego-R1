from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
import asyncio
import uuid
import os
import shutil
from video_llm import video_llm_with_client, clear_gemini # Assuming this is your module
from fastapi.middleware.cors import CORSMiddleware
import traceback
from google.genai.errors import ServerError
from concurrent.futures import ThreadPoolExecutor
import functools
from google import genai
import yaml

# Load configuration
def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Load configuration
config = load_config()
GEMINI_API_KEYS = config['egolife']['videollm']['gemini_api_keys']
DATA_DIR = config['egolife']['videollm']['data_dir']
GEMINI_PORT = config['egolife']['videollm']['gemini_port']

# Define a timeout for requests (e.g., 60 seconds)
REQUEST_TIMEOUT_SECONDS = 400

call_counter = 0
CLEAR_EVERY_N = 100
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

async def process_single_request_with_timeout(item: VideoLLMRequest, api_key: str) -> VideoLLMResponse:
    import random
    temp_cache_dir = f"./tmp/cache/task_{uuid.uuid4()}"
    os.makedirs(temp_cache_dir, exist_ok=True)

    tried_keys = set()
    last_exception = None
    try:
        for attempt in range(3): # Retry logic for ServerError 503
            try:
                tried_keys.add(api_key)
                print(f"Using API key: {api_key}")

                # Wrap the potentially long-running operation with asyncio.wait_for
                response = await asyncio.wait_for(
                    video_llm_with_client( # Assuming video_llm_with_client is an async function
                        question=item.question,
                        range_=item.range,
                        identity=item.identity,
                        cache_dir=temp_cache_dir,
                        gemini_api_key=api_key,
                        data_dir=DATA_DIR
                    ),
                    timeout=REQUEST_TIMEOUT_SECONDS
                )
                return VideoLLMResponse(**response)
            except asyncio.TimeoutError:
                print(f"Request timed out for item: {item.identity} with key {api_key}")
                raise # Re-raise to be caught by the main handler or batch processor
            except ServerError as e:
                if hasattr(e, 'code') and e.code == 503:
                    print(f"ServerError 503 encountered with key {api_key}, attempt {attempt+1}/3. Retrying with new key...")
                    last_exception = e
                    available_keys = [k for k in GEMINI_API_KEYS if k not in tried_keys]
                    if not available_keys:
                        available_keys = GEMINI_API_KEYS
                    if not available_keys: # Should not happen if GEMINI_API_KEYS is not empty
                        print("No API keys available to retry.")
                        raise e # No keys to try
                    api_key = random.choice(available_keys)
                    continue
                else:
                    raise # Re-raise other ServerErrors
            except Exception as e: # Catch other exceptions during the call
                last_exception = e
                break # Break retry loop for non-503 errors

        if last_exception:
            print(f"All retries failed or other exception occurred. Last exception: {last_exception}")
            raise last_exception # Re-raise the last encountered exception

        # This part should ideally not be reached if retries or the initial attempt succeeded or failed with an exception
        raise RuntimeError("process_single_request_with_timeout failed unexpectedly without a caught exception.")

    finally:
        shutil.rmtree(temp_cache_dir)

@app.post(
    "/video_llm",
    response_model=Union[VideoLLMResponse, str, List[Union[VideoLLMResponse, str, None]]]
)
async def video_llm_api(request: Union[VideoLLMRequest, List[Union[VideoLLMRequest, None, str]]]):
    import random
    global call_counter

    call_counter += 1
    print(f"[INFO] /video_llm API call count: {call_counter}")
    if call_counter % CLEAR_EVERY_N == 0:
        print(f"[INFO] Triggering clear_gemini after {CLEAR_EVERY_N} API calls.")
        if GEMINI_API_KEYS:
            temp_key = random.choice(GEMINI_API_KEYS)
            try:
                # Ensure genai.configure is called if you are using client directly without it
                # genai.configure(api_key=temp_key) # Or however your client is usually initialized
                video_client = genai.Client(api_key=temp_key) # If your clear_gemini expects a client
                clear_gemini(video_client) # Make sure clear_gemini can handle client or is parameterless
            except Exception as e:
                print(f"Error during clear_gemini: {e}")
        else:
            print("[WARN] No API keys available to initialize client for clear_gemini.")

    try:
        if isinstance(request, list):
            results_placeholder = [None] * len(request)
            tasks_to_run = []
            original_indices = []

            for idx, item_data in enumerate(request):
                if item_data is None or item_data == "" or (isinstance(item_data, str) and item_data.strip() == ""):
                    results_placeholder[idx] = None # Or "" or some other indicator for empty inputs
                    continue

                try:
                    if isinstance(item_data, dict):
                        item = VideoLLMRequest(**item_data)
                    elif isinstance(item_data, VideoLLMRequest):
                        item = item_data
                    else: # Handle invalid item types in the list
                        results_placeholder[idx] = "error: invalid item format in list"
                        continue

                    if not GEMINI_API_KEYS:
                        results_placeholder[idx] = "error: No API keys configured"
                        continue

                    api_key_for_task = random.choice(GEMINI_API_KEYS) # Assign key per task
                    # Create a partial that can be awaited
                    # process_single_request_with_timeout is already async
                    tasks_to_run.append(process_single_request_with_timeout(item, api_key_for_task))
                    original_indices.append(idx)

                except Exception as e: # Catch Pydantic validation errors or other setup issues
                    print(f"Error processing item at index {idx} before API call: {e}")
                    results_placeholder[idx] = f"error: {str(e)}"


            if tasks_to_run:
                # Use asyncio.gather to run tasks concurrently
                # return_exceptions=True allows us to get back exceptions instead of raising them immediately
                processed_results = await asyncio.gather(*tasks_to_run, return_exceptions=True)

                for i, result in enumerate(processed_results):
                    original_idx = original_indices[i]
                    if isinstance(result, asyncio.TimeoutError):
                        results_placeholder[original_idx] = "error: timeout"
                    elif isinstance(result, Exception):
                        results_placeholder[original_idx] = f"error: {str(result)}"
                    else:
                        results_placeholder[original_idx] = result
            print(results_placeholder)
            return results_placeholder
        else:
            # Single request
            if isinstance(request, dict):
                item = VideoLLMRequest(**request) # Validate and parse
            elif isinstance(request, VideoLLMRequest):
                item = request
            else:
                raise HTTPException(status_code=400, detail="Invalid request format for single item.")

            if not GEMINI_API_KEYS:
                return "error: No API keys configured"

            api_key = random.choice(GEMINI_API_KEYS)
            try:
                result = await process_single_request_with_timeout(item, api_key)
                return result
            except asyncio.TimeoutError:
                print(f"Single request timed out for item: {item.identity}")
                return "error: timeout"
            except Exception as e:
                print(f"Exception occurred in single request for item {item.identity}: {str(e)}")
                traceback.print_exc()
                return f"error: {str(e)}"

    except Exception as e: # Catch errors in the main API logic (e.g., request parsing for single item)
        print(f"Overall exception in video_llm_api: {str(e)}")
        traceback.print_exc()
        # Ensure a string is returned as per one of the Union types in response_model
        # For production, you might want a more structured error like ErrorResponse
        # but the current response_model allows str for errors.
        if isinstance(request, list):
             return [f"error: {str(e)}" for _ in request] # or a single error message string
        return f"error: {str(e)}"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=GEMINI_PORT)