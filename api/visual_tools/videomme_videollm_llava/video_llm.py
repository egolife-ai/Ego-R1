# videollm.py
from google import genai
from typing_extensions import Annotated
import time
import os
import ffmpeg
from utils import locate_videos, config_videos
import contextlib
import requests



def video_llm(question: str, range: str,  cache_dir: str, data_dir: str) -> dict:
    try:
        start_time, end_time = range.split("-")
    except:
        raise ValueError("Invalid range format. Use DAYX_HHMMSSFF-DAYX_HHMMSSFF.")

   

    

    file_paths, exact_match,start_time = locate_videos(start_time, end_time, data_dir=data_dir, cache_dir=cache_dir)
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



