import os
from datetime import datetime
from typing_extensions import Annotated
import random
import requests
from google import genai
from utils import *
import time
import asyncio
import ffmpeg

rag_dict = {
    "A1_JAKE": "http://localhost:8001/query",
    "A2_ALICE": "http://localhost:8002/query",
    "A3_TASHA": "http://localhost:8003/query",
    "A4_LUCIA": "http://localhost:8004/query",
    "A5_KATRINA": "http://localhost:8005/query",
    "A6_SHURE": "http://localhost:8006/query"
}

CLEAR_GEMINI_TIME = time.time()
gemini_api_key = os.getenv("GEMINI_API_KEY") if os.getenv("GEMINI_API_KEY") else model_config["Gemini_API_KEY"]
video_client = genai.Client(api_key=gemini_api_key)

def clear_gemini(client: genai.Client):

    for f in client.files.list():
        print(' ', f.name)
        try:
            client.files.delete(name=f.name)
            print(f"Deleted file: {f.name}")
        except Exception as e:
            print(f"Failed to delete {f.name}: {e}")

    print("All files deleted!")

def clear_cache():

    for f in os.listdir(os.environ["CACHE_DIR"]):
        os.remove(os.path.join(os.environ["CACHE_DIR"], f))



async def rag(level: Annotated[str, "The granularity of the search, choose from week|day|hour"], keywords: Annotated[list[str], "The keywords to search for as a reference in database"], start_time: Annotated[str, "The timestamp of the start time of the search. The format should be DAYX_HHMMSSFF (X is the day number, HHMMSS is the hour, minute, second, and FF is the frame number(00~19))"], query_time: Annotated[str, "The timestamp of the query that was proposed by the user. The format should be DAYX_HHMMSSFF (X is the day number, HHMMSS is the hour, minute, second, and FF is the frame number(00~19))"]) -> dict:
    """
    This function is the most critical and powerful tool for retrieving information across a large temporal range from the database. It should be your first choice for almost every question. It provides access to rich contextual data across different time granularities (week/day/hour), making it essential for answering questions about past events, timelines, and relationships between activities. Using this tool multiple times with different keywords and time ranges is highly recommended to thoroughly explore the problem space before attempting other approaches.
    
    FREQUENCY: 6/10
    COST: 1/5
    TEMPORAL CAPABILITY: 5/5
    VISUAL INPUT: False
    
    ## HINT
    1. Use as first choice. For uncertain cases, follow hierarchical search: weekly logs → daily logs → hourly logs.
    2. If direct search fails, try related terms (people, events, objects) associated with the query topic.
    """
    if os.environ["RAG_URL"] is not None:
        url = os.environ["RAG_URL"]
    else:
        url = "http://localhost:8001/query"
    response = requests.post(
        url,
        json={
            "level": level,
            "keywords": keywords,
            "start_time": start_time,
            "query_time": query_time
        }
    )
    return response.json()

async def rag_downtop(level: Annotated[str, "The granularity of the search, choose from week|day|hour"], keywords: Annotated[list[str], "The keywords to search for as a reference in database"], start_time: Annotated[str, "The timestamp of the start time of the search. The format should be DAYX_HHMMSSFF (X is the day number, HHMMSS is the hour, minute, second, and FF is the frame number(00~19))"], query_time: Annotated[str, "The timestamp of the query that was proposed by the user. The format should be DAYX_HHMMSSFF (X is the day number, HHMMSS is the hour, minute, second, and FF is the frame number(00~19))"]) -> dict:
    """
    This is the similarity-based search. This function is the most critical and powerful tool for retrieving information across a large temporal range from the database. It should be your first choice for almost every question. It provides access to rich contextual data across different time granularities (week/day/hour), making it essential for answering questions about past events, timelines, and relationships between activities. Using this tool multiple times with different keywords and time ranges is highly recommended to thoroughly explore the problem space before attempting other approaches.
    
    FREQUENCY: 6/10
    COST: 1/5
    TEMPORAL CAPABILITY: 5/5
    VISUAL INPUT: False
    
    ## HINT
    1. Use as first choice. For uncertain cases, follow hierarchical search: weekly logs → daily logs → hourly logs.
    2. If direct search fails, try related terms (people, events, objects) associated with the query topic.
    """
    response = requests.post(
    "http://localhost:8008/query",

        json={
            "level": level,
            "keywords": keywords,
            "start_time": start_time,
            "query_time": query_time
        }
    )
    return response.json()


async def rag_simulated(level: Annotated[str, "The granularity of the search, choose from day|hour|minute"], keywords: Annotated[list[str], "The keywords to search for as a reference in database"], start_time: Annotated[str, "The start time of the search"], query_time: Annotated[str, "The time of the query that was proposed by the user"]) -> dict:
    """
    {’date’: ’1’, ’start_time’: ’11090000’, ’relevant_content’: ’Once marked, I passed the phone around the table for others to mark it too, ensuring everyone participated. After confirming the task was complete, I turned off the phone and prepared for a meeting. ’}
    
    """
    responses = [
        {
            "date": "1",
            "start_time": "11090000",
            "relevant_content": "Once marked, I passed the phone around the table for others to mark it too, ensuring everyone participated. After confirming the task was complete, I turned off the phone and prepared for a meeting."
        },
        {
            "date": "3",
            "start_time": "12090000",
            "relevant_content": "I went to the supermarket to buy groceries and shopping. I bought some fruits and vegetables."
        },
        {
            "date": "2",
            "start_time": "13090000",
            "relevant_content": "I collaborated with my friends to clean the house. We also had a nice dinner together."
        },
        {
            "level": "day",
            "date": "1",
            "start_time": "12015000",
            "relevant_content": "I put my phone on the table when we arrived at the house. Alice and Tasha kept their phones in their bags."
        }
    ]
    response = random.choice(responses)
    return response

async def video_llm(question: Annotated[str, "The question you want to use the video language model to answer. Note: The question does not necessarily need to be the same as the question given by the user. You should propose a question based on the available observations."], range: Annotated[str, "The timestamp range of the video to answer the question. The format should be DAYX_HHMMSSFF-DAYX_HHMMSSFF (X is the day number, HHMMSS is the hour, minute, second, and FF is the frame number(00~19))."]) -> dict:
    """
    Analyze video content at a specific timestamp using a video language model.
    Answers questions about visual scenes, events, and objects.
    Returns both the answer and video file metadata.
    
    FREQUENCY: 3/10
    COST: 3/5
    TEMPORAL CAPABILITY: 4/5
    VISUAL INPUT: True
    
    ## HINT
    1. Ask targeted questions that align with your observation and reasoning process rather than repeating the user's original question.
    2. Longer range will be more costy.
    3. The maximum query range is 10 minutes. The recommended query range is from 30 seconds to 10 minutes.
    """
    try:
        start_time, end_time = range.split("-")
    except:
        raise ValueError("Invalid range format. Please use the format DAYX_HHMMSSFF-DAYX_HHMMSSFF (X is the day number, HHMMSS is the hour, minute, second, and FF is the frame number(00~19)).")
    # file_path, exact_match = locate_video_url(start_time)
    # clear gemini cache periodically
    global CLEAR_GEMINI_TIME
    global video_client
    
    current_time = time.time()
    if current_time - CLEAR_GEMINI_TIME > 300:  # Clear cache every 5 minutes
        clear_gemini(video_client)
        CLEAR_GEMINI_TIME = current_time
    clear_cache()
    file_paths, exact_match = locate_videos(start_time, end_time, identity=os.environ["IDENTITY"])
    
    # trim * combine the videos
    combined_video, length = config_videos(file_paths, start_time, end_time)
    
    print(f"Uploading video from {combined_video} to Gemini...")
    video_file = video_client.files.upload(file=combined_video)
    print(f"Completed upload: {video_file.uri}")
    
    while video_file.state.name == "PROCESSING":
        print('.', end='')
        time.sleep(1)
        video_file = video_client.files.get(name=video_file.name)
    
    if video_file.state.name == "FAILED":
        raise ValueError(video_file.state.name)

    print('Done')
    
    
    # Pass the video file reference like any other media part.
    response = video_client.models.generate_content(
        model="gemini-1.5-pro",
        contents=[video_file, question]
    )

    # Print the response, rendering any Markdown
    print(response.text)
    
    response = {
        "query_time": range,
        "question": question,
        "answer": response.text,
        "exact_match": exact_match,
        "length": length
    }
    # Markdown(response.text)
    return response


async def vlm(question: Annotated[str, "The question you want to use the vision language model to answer. Note: The question does not necessarily need to be the same as the question given by the user. You should propose a question based on the available observations."], timestamp: Annotated[str, "The timestamp of the video to answer the question. The format should be DAYX_HHMMSSFF (X is the day number, HHMMSS is the hour, minute, second, and FF is the frame number(00~19))"]) -> dict:
    """
    Analyze a single image frame at the specified timestamp using a vision language model.
    Answers questions about visual scenes, events, and objects.
    Returns both the answer and image file metadata.
    
    FREQUENCY: 2/10    
    COST: 2/5
    TEMPORAL CAPABILITY: 1/5
    VISUAL INPUT: True
    
    ## HINT
    1. Ask targeted questions that align with your observation and reasoning process rather than repeating the user's original question.
    """
    
    # Use OpenAI API to answer the question
    os.makedirs(os.environ["CACHE_DIR"], exist_ok=True)
    frame_path, exact_match = locate_image_url(timestamp, identity=os.environ["IDENTITY"])
    
    image_client = GPT()
    answer = image_client.chat(question, frame_path)
    
    response = {
        "query_time": timestamp,
        "question": question,
        "answer": answer,
        "exact_match": exact_match
    }
    return response


async def terminate(answer: Annotated[str, "The answer to the question. Choose from A|B|C|D. DISCLAIMER: DO NOT USE THIS TOOL UNLESS YOU HAVE TRIED ALL THE OTHER TOOLS AND STILL CANNOT ANSWER THE QUESTION. If you cannot answer the question, take a deep breath and think about using other tools to help with the question. Do not give up easily unless you have tried all the tools and still cannot answer the question. After you have tried all the tools and still cannot answer the question, return 'N/A'."]) -> dict:
    """
    Terminate the conversation and return the answer.
    
    FREQUENCY: 1/10 (once per task)
    
    ## HINT
    1. Only use this tool once you have tried all the other tools and still cannot answer the question.
    2. If you cannot answer the question, take a deep breath and think about using other tools to help with the question. Do not give up easily unless you have tried all the tools and still cannot answer the question.
    3. If you have already tried all the tools and still cannot answer the question, return 'N/A'.
    """
    return {"answer": answer}

async def terminate_explicit(answer: Annotated[str, "The answer to the question. Provide the free-form answer. The answer should be concise and to the point. DISCLAIMER: DO NOT USE THIS TOOL UNLESS YOU HAVE TRIED ALL THE OTHER TOOLS AND STILL CANNOT ANSWER THE QUESTION. If you cannot answer the question, take a deep breath and think about using other tools to help with the question. Do not give up easily unless you have tried all the tools and still cannot answer the question. After you have tried all the tools and still cannot answer the question, return 'N/A'."]) -> dict:
    """
    Terminate the conversation and return the answer.
    
    FREQUENCY: 1/10 (once per task)
    
    ## HINT
    1. Only use this tool once you have tried all the other tools and still cannot answer the question.
    2. If you cannot answer the question, take a deep breath and think about using other tools to help with the question. Do not give up easily unless you have tried all the tools and still cannot answer the question.
    3. If you have already tried all the tools and still cannot answer the question, return 'N/A'.
    """
    return {"answer": answer}
