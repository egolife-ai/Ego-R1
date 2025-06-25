# videollm.py
from google import genai
from typing_extensions import Annotated
import time
import os
import ffmpeg
from utils import locate_videos, config_videos
import contextlib
import requests



@contextlib.contextmanager
def override_cache_dir(new_cache_dir: str):
    """Temporarily override the CACHE_DIR environment variable."""
    old_cache_dir = os.environ.get("CACHE_DIR")
    os.environ["CACHE_DIR"] = new_cache_dir
    try:
        yield
    finally:
        if old_cache_dir is not None:
            os.environ["CACHE_DIR"] = old_cache_dir

def video_llm_with_client(question: str, range_: str, identity: str, cache_dir: str, gemini_api_key: str, data_dir: str) -> dict:
    return video_llm(question=question, range=range_, identity=identity, cache_dir=cache_dir, data_dir=data_dir)

def video_llm(question: str, range: str, identity: str, cache_dir: str, data_dir: str) -> dict:
    try:
        start_time, end_time = range.split("-")
    except:
        raise ValueError("Invalid range format. Use DAYX_HHMMSSFF-DAYX_HHMMSSFF.")

   

    file_paths, exact_match = locate_videos(start_time, end_time, identity=identity, cache_dir=cache_dir, data_dir=data_dir)
    combined_video, length = config_videos(file_paths, start_time, end_time, cache_dir=cache_dir)
    
    url = "http://localhost:8000/video/infer"
    pay_load = {
        "video_path": combined_video,
        "prompt": question,
        "max_frames": 64,
        "frame_step": 1
    }
    
    
    response = requests.post(url, json=pay_load)

    print(response.json()['response'])

    return {
        "query_time": range,
        "question": question,
        "answer": response.json()['response'],
        "exact_match": exact_match,
        "length": length
    }

