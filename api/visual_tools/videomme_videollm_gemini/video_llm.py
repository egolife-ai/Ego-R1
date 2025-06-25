# videollm.py
from google import genai
from typing_extensions import Annotated
import time
import os
import ffmpeg
from utils import locate_videos, config_videos
import contextlib

# 需要从外部设置 model_config["Gemini_API_KEY"]
model_config = {
    "Gemini_API_KEY": None
}

def get_gemini_client():
    """Create a new Gemini client instance"""
    return genai.Client(api_key=model_config["Gemini_API_KEY"])

def get_gemini_client_with_key(api_key: str):
    """Create a Gemini client with a specific API Key"""
    return genai.Client(api_key=api_key)

def clear_gemini(client: genai.Client):
    """Clear all uploaded files on Gemini server"""
    files = list(client.files.list())
    if not files:
        print("No files to delete.")
        return

    for f in files:
        try:
            if f.name:
                client.files.delete(name=f.name)
                print(f"Deleted file: {f.name}")
        except Exception as e:
            print(f"Failed to delete {f.name}: {e}")
    print("All files deleted!")

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

def video_llm_with_client(question: str, range_: str, cache_dir: str, gemini_api_key: str, data_dir: str) -> dict:
   
    return video_llm(question=question, range=range_, cache_dir=cache_dir, data_dir=data_dir,api_key=gemini_api_key)

def video_llm(question: str, range: str,  cache_dir: str, data_dir: str, api_key: str) -> dict:
    try:
        start_time, end_time = range.split("-")
    except:
        raise ValueError("Invalid range format. Use DAYX_HHMMSSFF-DAYX_HHMMSSFF.")

    video_client = genai.Client(api_key=api_key)
   

    clear_cache(cache_dir)

    file_paths, exact_match,start_time = locate_videos(start_time, end_time, data_dir=data_dir, cache_dir=cache_dir)
    combined_video, length = config_videos(file_paths, start_time, end_time, cache_dir=cache_dir)


    print(f"Uploading video from {combined_video} to Gemini...")
    video_file = video_client.files.upload(file=combined_video)
    print(f"Completed upload: {video_file.uri}")

    if video_file and getattr(video_file, 'state', None):
        while getattr(video_file.state, 'name', None) == "PROCESSING":
            print('.', end='')
            time.sleep(1)
            video_file = video_client.files.get(name=video_file.name) if getattr(video_file, 'name', None) else None

        if getattr(video_file.state, 'name', None) == "FAILED":
            raise ValueError(getattr(video_file.state, 'name', None))

    print('Done')

    response = video_client.models.generate_content(
        model="gemini-1.5-pro",
        contents=[video_file, question]
    )

    print(response.text)

    return {
        "query_time": range,
        "question": question,
        "answer": response.text,
        "exact_match": exact_match,
        "length": length
    }

def clear_cache(cache_dir=None):
    if cache_dir and os.path.exists(cache_dir):
        for f in os.listdir(cache_dir):
            os.remove(os.path.join(cache_dir, f))
