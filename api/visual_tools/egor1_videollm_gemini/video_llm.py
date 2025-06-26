# videollm.py
from google import genai
import time
import os
import ffmpeg # Assuming utils.config_videos indirectly or directly uses ffmpeg-python
from utils import locate_videos, config_videos # Assuming these exist and are synchronous
import asyncio
import functools # For functools.partial
import shutil # For shutil.rmtree

def clear_gemini(client: genai.Client):
    """Clear all uploaded files from Gemini server"""
    try:
        files = list(client.files.list()) # This might also be a blocking network call
        if not files:
            print("No files to delete.")
            return

        deleted_count = 0
        failed_count = 0
        for f in files:
            try:
                if hasattr(f, 'name') and f.name: # Check if f.name exists and is not empty
                    print(f"Attempting to delete file: {f.name}")
                    client.files.delete(name=f.name) # Blocking network call
                    print(f"Deleted file: {f.name}")
                    deleted_count += 1
                else:
                    print(f"Skipping a file without a valid name for deletion.")
            except Exception as e:
                failed_count +=1
                file_identifier = f.name if hasattr(f, 'name') and f.name else 'an unnamed file'
                print(f"Failed to delete {file_identifier}: {e}")
        print(f"File deletion process completed! Successfully deleted {deleted_count} files, failed {failed_count}.")
    except Exception as e:
        print(f"Error while listing files to delete: {e}")


# This is the synchronous blocking function that performs the core work
def video_llm_sync(question: str, range_str: str, identity: str, cache_dir: str, api_key: str,data_dir:str) -> dict:
    """
    Synchronously process video LLM logic: locate video, configure, upload to Gemini, get analysis results.
    This is a blocking function that should be run in a separate thread to avoid blocking the asyncio event loop.
    """
    try:
        start_time_str, end_time_str = range_str.split("-")
    except ValueError:
        raise ValueError("Invalid range format. Please use DAYX_HHMMSSFF-DAYX_HHMMSSFF format.")

    video_client = genai.Client(api_key=api_key)
    print(f"Using API Key: ...{api_key[-4:]} for video_llm_sync processing (Identity: {identity})")

    clear_cache(cache_dir) # Synchronous file operations

    file_paths, exact_match = locate_videos(start_time_str, end_time_str, identity=identity, cache_dir=cache_dir,data_dir=data_dir) # Synchronous
    if not file_paths:
        raise FileNotFoundError(f"No video files found for identity {identity} within range {range_str}.")

    combined_video, length = config_videos(file_paths, start_time_str, end_time_str, cache_dir=cache_dir) # Synchronous, potentially time-consuming (ffmpeg)
    if not combined_video or not os.path.exists(combined_video):
        raise FileNotFoundError(f"Failed to create combined video from {file_paths}.")

    print(f"Uploading video from {combined_video} to Gemini... (Identity: {identity})")
    upload_start_time = time.time()
    video_file = None # Initialize
    try:
        video_file = video_client.files.upload(file=combined_video)
        print(f"Upload call completed: {video_file.uri}, current status: {getattr(getattr(video_file, 'state', None), 'name', 'unknown')} (Identity: {identity})")

        polling_start_time = time.time()
        max_polling_time_seconds = 300 # Internal polling timeout for Gemini file processing (e.g., 8 minutes)

        while getattr(video_file, 'state', None) and getattr(video_file.state, 'name', None) == "PROCESSING":
            if time.time() - polling_start_time > max_polling_time_seconds:
                raise TimeoutError(f"Gemini file processing timed out for {video_file.name} ({max_polling_time_seconds} seconds). (Identity: {identity})")

            print(f'. (Processing {video_file.name} for identity {identity})', end='', flush=True)
            time.sleep(10) # Polling interval
            try:
                video_file = video_client.files.get(name=video_file.name)
            except Exception as e:
                raise RuntimeError(f"Error getting status for file {video_file.name}: {e} (Identity: {identity})")

        if not (getattr(video_file, 'state', None) and getattr(video_file.state, 'name', 'UNKNOWN') == "ACTIVE"):
            current_state = getattr(getattr(video_file, 'state', None), 'name', 'unknown')
            failure_reason_obj = getattr(video_file.state, 'failure_reason', None)
            failure_details = getattr(failure_reason_obj, 'name', 'no specific reason') if failure_reason_obj else 'no specific reason'
            # Special handling for file processing failures due to prohibited content
            if current_state == "FAILED" and "PROHIBITED_CONTENT" in failure_details:
                 raise ValueError(f"Video file {video_file.name} processing failed due to prohibited content. Status: {current_state}, Reason: {failure_details}. (Identity: {identity})")
            raise RuntimeError(f"Video file {video_file.name} not activated after processing. Current status: {current_state}, Reason: {failure_details}. (Identity: {identity})")

        upload_and_processing_time = time.time() - upload_start_time
        print(f'\nCompleted. Upload and processing of {video_file.name} took {upload_and_processing_time:.2f} seconds. (Identity: {identity})')

        model_to_use = "gemini-1.5-pro-latest"
        print(f"Using model: {model_to_use} to generate content for file: {video_file.uri}... (Identity: {identity})")

       
        response = video_client.models.generate_content(model="gemini-1.5-pro", contents=[video_file, question])
        
        answer_text = ""
        if response.text:
            answer_text = response.text
        else: # If there's no text part, log the completion reason
            try:
                finish_reason = response.candidates[0].finish_reason if response.candidates else "unknown reason"
                category = response.candidates[0].safety_ratings[0].category if response.candidates and response.candidates[0].safety_ratings else "unknown category"
                probability = response.candidates[0].safety_ratings[0].probability if response.candidates and response.candidates[0].safety_ratings else "unknown probability"
                print(f"Warning: No text in response. Completion reason: {finish_reason}, Safety category: {category} ({probability}) for identity: {identity}")
                if not isinstance(finish_reason, str) and hasattr(finish_reason, 'name') and finish_reason.name == "SAFETY": # In Python SDK, finish_reason is an enum
                    raise ValueError(f"Content generation stopped due to safety reasons (Identity: {identity})")
            except IndexError:
                 print(f"Warning: No text in response, and unable to get completion reason or safety rating. (Identity: {identity})")


        print(f"Question: {question} for Identity: {identity} -> Answer: {answer_text[:50]}...")

        return {
            "query_time": range_str,
            "question": question,
            "answer": answer_text,
            "exact_match": exact_match,
            "length": length
        }
    except Exception as e:
        print(f"Error: {e}")
        raise e


# This is the async wrapper that runs the sync function in a thread pool
async def video_llm_with_client(question: str, range_: str, identity: str, cache_dir: str, gemini_api_key: str,data_dir:str) -> dict:
    """
    Asynchronously call the video_llm_sync function.
    It uses asyncio's run_in_executor to dispatch blocking code to a separate thread.
    """
    loop = asyncio.get_event_loop()
    # Use functools.partial to pass parameters to video_llm_sync that will be called in the executor
    func_call = functools.partial(video_llm_sync, question, range_, identity, cache_dir, gemini_api_key,data_dir)
    
    # Run the synchronous function in the default thread pool executor
    result_dict = await loop.run_in_executor(None, func_call)
    return result_dict


def clear_cache(cache_dir=None):
    """Clear all files from the local cache directory."""
    if cache_dir and os.path.exists(cache_dir):
        if os.path.isdir(cache_dir):
            print(f"Clearing cache directory: {cache_dir}")
            # shutil.rmtree(cache_dir) # Will delete the entire directory
            # os.makedirs(cache_dir, exist_ok=True) # Then rebuild it
            # Or just delete the contents:
            for f_name in os.listdir(cache_dir):
                f_path = os.path.join(cache_dir, f_name)
                try:
                    if os.path.isfile(f_path) or os.path.islink(f_path):
                        os.remove(f_path)
                    elif os.path.isdir(f_path):
                        shutil.rmtree(f_path) # If there might be subdirectories in cache, delete recursively
                except Exception as e:
                    print(f"Failed to delete {f_path} from cache: {e}")
            print(f"Cache directory {cache_dir} cleared.")

        else:
            print(f"Warning: cache_dir '{cache_dir}' is not a directory.")